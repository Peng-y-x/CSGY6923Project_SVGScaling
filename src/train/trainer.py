from __future__ import annotations

import json
import math
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import torch

from src.models.transformer import DecoderOnlyTransformer, TransformerConfig
from src.train.metrics import append_jsonl, write_json
from src.train.optim import build_adamw
from src.train.schedulers import build_cosine_warmup_scheduler


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class TokenizedSvgDataset:
    def __init__(self, cfg: dict[str, Any], split: str) -> None:
        self.cfg = cfg
        self.split = split
        source = cfg.get("source", "hf")
        if source == "hf":
            try:
                from datasets import load_dataset
            except ImportError as exc:  # pragma: no cover
                raise ImportError("Install datasets to load tokenized HF data.") from exc
            repo_id = cfg["hf_repo_id"]
            self.ds = load_dataset(repo_id, split=split)
        elif source == "jsonl":
            path = Path(cfg["paths"][split])
            rows = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    rows.append(json.loads(line))
            self.ds = rows
        else:
            raise ValueError(f"Unsupported data.source: {source}")

    def __len__(self) -> int:
        return len(self.ds)

    def ids_at(self, index: int) -> list[int]:
        row = self.ds[index]
        ids = row["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return [int(x) for x in ids]

    def iter_batches(
        self,
        *,
        start_index: int,
        tokens_per_batch: int,
        block_size: int,
        pad_token_id: int,
        max_tokens: int | None = None,
    ) -> Iterator[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, int]]:
        batch: list[list[int]] = []
        batch_tokens = 0
        emitted_tokens = 0
        next_index = start_index

        for idx in range(start_index, len(self)):
            ids = self.ids_at(idx)
            if len(ids) < 2:
                next_index = idx + 1
                continue
            ids = ids[: block_size + 1]
            seq_tokens = len(ids) - 1
            if batch and batch_tokens + seq_tokens > tokens_per_batch:
                x, y, mask, ntok = _collate_lm_batch(batch, pad_token_id)
                emitted_tokens += ntok
                yield idx, x, y, mask, ntok
                batch = []
                batch_tokens = 0
                if max_tokens is not None and emitted_tokens >= max_tokens:
                    return

            batch.append(ids)
            batch_tokens += seq_tokens
            next_index = idx + 1

        if batch:
            x, y, mask, ntok = _collate_lm_batch(batch, pad_token_id)
            yield next_index, x, y, mask, ntok


def _collate_lm_batch(rows: list[list[int]], pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    max_len = max(len(x) for x in rows) - 1
    input_ids = torch.full((len(rows), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(rows), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long)

    for row_idx, ids in enumerate(rows):
        x = ids[:-1]
        y = ids[1:]
        seq_len = len(x)
        input_ids[row_idx, :seq_len] = torch.tensor(x, dtype=torch.long)
        labels[row_idx, :seq_len] = torch.tensor(y, dtype=torch.long)
        attention_mask[row_idx, :seq_len] = 1

    return input_ids, labels, attention_mask, int((labels != -100).sum().item())


def _expand_colab_path(path: str | None) -> Path | None:
    if not path:
        return None
    return Path(os.path.expandvars(os.path.expanduser(path)))


def _sync_run_dir(local_run_dir: Path, drive_run_dir: Path) -> None:
    drive_run_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(local_run_dir, drive_run_dir, dirs_exist_ok=True)


class Part2Trainer:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        run_cfg = cfg["run"]
        self.run_name = run_cfg["name"]
        self.run_dir = Path(run_cfg.get("output_dir", "outputs/part2")) / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.drive_run_dir = None
        drive_dir = _expand_colab_path(run_cfg.get("drive_output_dir"))
        if drive_dir is not None:
            self.drive_run_dir = drive_dir / self.run_name

        seed = int(cfg.get("seed", 42))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        device_name = run_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
        self.precision = str(run_cfg.get("precision", "bf16"))
        self.autocast_dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        self.use_autocast = self.device.type == "cuda" and self.precision in {"bf16", "fp16"}
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda" and self.precision == "fp16")

        model_cfg = cfg["model"]
        self.model_config = TransformerConfig(**model_cfg)
        self.model = DecoderOnlyTransformer(self.model_config).to(self.device)

        train_cfg = cfg["training"]
        self.optimizer = build_adamw(
            self.model,
            learning_rate=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg.get("weight_decay", 0.1)),
            betas=tuple(train_cfg.get("betas", [0.9, 0.95])),
            eps=float(train_cfg.get("eps", 1e-8)),
        )

        self.train_data = TokenizedSvgDataset(cfg["data"], "train")
        self.val_data = TokenizedSvgDataset(cfg["data"], "validation")

        total_steps = self._estimate_total_steps()
        warmup_steps = int(train_cfg.get("warmup_steps", 0))
        if warmup_steps <= 0:
            warmup_steps = int(total_steps * float(train_cfg.get("warmup_ratio", 0.02)))
        self.scheduler = build_cosine_warmup_scheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=float(train_cfg.get("min_lr_ratio", 0.1)),
        )

        self.global_step = 0
        self.tokens_seen = 0
        self.next_row_index = 0
        self.best_val_loss = float("inf")
        self.started_at = time.time()

    def _estimate_total_steps(self) -> int:
        data_cfg = self.cfg["data"]
        train_cfg = self.cfg["training"]
        train_tokens = int(data_cfg.get("train_total_tokens", 0))
        max_train_tokens = int(train_cfg.get("max_train_tokens", 0))
        if max_train_tokens > 0:
            train_tokens = min(train_tokens or max_train_tokens, max_train_tokens)
        if train_tokens <= 0:
            train_tokens = len(self.train_data) * int(data_cfg.get("mean_seq_len", 512))
        return max(1, math.ceil(train_tokens / int(train_cfg["tokens_per_batch"])))

    def _checkpoint_payload(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "global_step": self.global_step,
            "tokens_seen": self.tokens_seen,
            "next_row_index": self.next_row_index,
            "best_val_loss": self.best_val_loss,
            "config": self.cfg,
            "model_config": asdict(self.model_config),
        }

    def _latest_checkpoint_path(self, base: Path) -> Path:
        return base / "checkpoints" / "latest.pt"

    def maybe_resume(self) -> None:
        resume = str(self.cfg["run"].get("resume", "auto"))
        if resume.lower() in {"false", "none", "off"}:
            return

        candidates: list[Path] = []
        explicit = self.cfg["run"].get("resume_from")
        if explicit:
            candidates.append(Path(explicit))
        candidates.append(self._latest_checkpoint_path(self.run_dir))
        if self.drive_run_dir is not None:
            candidates.append(self._latest_checkpoint_path(self.drive_run_dir))

        ckpt_path = next((p for p in candidates if p.exists()), None)
        if ckpt_path is None:
            return

        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])
        self.global_step = int(ckpt.get("global_step", 0))
        self.tokens_seen = int(ckpt.get("tokens_seen", 0))
        self.next_row_index = int(ckpt.get("next_row_index", 0))
        self.best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        print(f"[resume] Loaded checkpoint: {ckpt_path}")
        print(f"[resume] step={self.global_step} tokens={self.tokens_seen} next_row={self.next_row_index}")

    def save_checkpoint(self, *, is_best: bool = False) -> None:
        ckpt_dir = self.run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        payload = self._checkpoint_payload()
        latest_path = ckpt_dir / "latest.pt"
        torch.save(payload, latest_path)
        if is_best:
            shutil.copy2(latest_path, ckpt_dir / "best.pt")
        if not bool(self.cfg["run"].get("save_latest_only", True)):
            shutil.copy2(latest_path, ckpt_dir / f"step_{self.global_step:07d}.pt")

        write_json(
            self.run_dir / "summary.json",
            {
                "run_name": self.run_name,
                "global_step": self.global_step,
                "tokens_seen": self.tokens_seen,
                "next_row_index": self.next_row_index,
                "best_val_loss": self.best_val_loss,
                "num_parameters": self.model.num_parameters(),
                "num_parameters_non_embedding": self.model.num_parameters(non_embedding=True),
                "elapsed_seconds": time.time() - self.started_at,
            },
        )

        if self.drive_run_dir is not None:
            _sync_run_dir(self.run_dir, self.drive_run_dir)
            print(f"[checkpoint] Synced to Drive: {self.drive_run_dir}")

    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        train_cfg = self.cfg["training"]
        max_eval_tokens = int(train_cfg.get("max_eval_tokens", 0)) or None
        losses = []
        total_tokens = 0
        with torch.no_grad():
            for _, x, y, mask, ntok in self.val_data.iter_batches(
                start_index=0,
                tokens_per_batch=int(train_cfg["tokens_per_batch"]),
                block_size=int(self.model_config.block_size),
                pad_token_id=int(self.model_config.pad_token_id),
                max_tokens=max_eval_tokens,
            ):
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)
                with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.use_autocast):
                    _, loss = self.model(x, labels=y, attention_mask=mask)
                assert loss is not None
                losses.append(float(loss.item()) * ntok)
                total_tokens += ntok
        val_loss = sum(losses) / max(1, total_tokens)
        self.model.train()
        return {
            "val_loss": val_loss,
            "val_ppl": math.exp(min(20.0, val_loss)),
            "val_tokens": float(total_tokens),
        }

    def train_one_epoch(self) -> dict[str, Any]:
        self.maybe_resume()
        self.model.train()
        train_cfg = self.cfg["training"]
        log_interval = int(train_cfg.get("log_interval_steps", 20))
        eval_interval = int(train_cfg.get("eval_interval_steps", 0))
        checkpoint_interval = int(train_cfg.get("checkpoint_interval_steps", 200))
        max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
        max_train_tokens = int(train_cfg.get("max_train_tokens", 0)) or None
        metrics_path = self.run_dir / "metrics.jsonl"

        if self.next_row_index >= len(self.train_data):
            print("[train] Checkpoint is already at or past the end of the train split.")
            final_metrics = self.evaluate()
            final_metrics.update(self._final_summary_fields())
            write_json(self.run_dir / "final_metrics.json", final_metrics)
            self.save_checkpoint()
            return final_metrics

        step_start = time.time()
        recent_loss = 0.0
        recent_tokens = 0
        for next_row, x, y, mask, ntok in self.train_data.iter_batches(
            start_index=self.next_row_index,
            tokens_per_batch=int(train_cfg["tokens_per_batch"]),
            block_size=int(self.model_config.block_size),
            pad_token_id=int(self.model_config.pad_token_id),
            max_tokens=max_train_tokens,
        ):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.use_autocast):
                _, loss = self.model(x, labels=y, attention_mask=mask)
            assert loss is not None

            self.scaler.scale(loss).backward()
            if max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            self.global_step += 1
            self.tokens_seen += ntok
            self.next_row_index = next_row
            recent_loss += float(loss.item()) * ntok
            recent_tokens += ntok

            if self.global_step % log_interval == 0:
                elapsed = max(1e-6, time.time() - step_start)
                row = {
                    "type": "train",
                    "step": self.global_step,
                    "tokens_seen": self.tokens_seen,
                    "next_row_index": self.next_row_index,
                    "train_loss": recent_loss / max(1, recent_tokens),
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "tokens_per_second": recent_tokens / elapsed,
                    "gpu_memory_gb": self._gpu_memory_gb(),
                    "elapsed_seconds": time.time() - self.started_at,
                }
                append_jsonl(metrics_path, row)
                print(json.dumps(row, ensure_ascii=False))
                recent_loss = 0.0
                recent_tokens = 0
                step_start = time.time()

            if eval_interval > 0 and self.global_step % eval_interval == 0:
                metrics = self.evaluate()
                metrics.update({"type": "eval", "step": self.global_step, "tokens_seen": self.tokens_seen})
                append_jsonl(metrics_path, metrics)
                is_best = metrics["val_loss"] < self.best_val_loss
                if is_best:
                    self.best_val_loss = metrics["val_loss"]
                self.save_checkpoint(is_best=is_best)

            if checkpoint_interval > 0 and self.global_step % checkpoint_interval == 0:
                self.save_checkpoint()

        final_metrics = self.evaluate()
        final_metrics.update(self._final_summary_fields())
        write_json(self.run_dir / "final_metrics.json", final_metrics)
        append_jsonl(metrics_path, {"type": "final", **final_metrics})
        is_best = final_metrics["val_loss"] < self.best_val_loss
        if is_best:
            self.best_val_loss = final_metrics["val_loss"]
        self.save_checkpoint(is_best=is_best)
        return final_metrics

    def _final_summary_fields(self) -> dict[str, Any]:
        elapsed = time.time() - self.started_at
        return {
            "run_name": self.run_name,
            "global_step": self.global_step,
            "tokens_seen": self.tokens_seen,
            "num_parameters": self.model.num_parameters(),
            "num_parameters_non_embedding": self.model.num_parameters(non_embedding=True),
            "wall_clock_seconds": elapsed,
            "tokens_per_second_epoch": self.tokens_seen / max(1e-6, elapsed),
            "peak_gpu_memory_gb": self._gpu_memory_gb(),
        }

    def _gpu_memory_gb(self) -> float:
        if self.device.type != "cuda":
            return 0.0
        return float(torch.cuda.max_memory_allocated(self.device) / (1024**3))


class Part3MupTrainer(Part2Trainer):
    def __init__(self, cfg: dict[str, Any]) -> None:
        from mup import MuAdamW, set_base_shapes

        from src.models.mup_transformer import (
            MupDecoderOnlyTransformer,
            MupTransformerConfig,
            make_mup_base_config,
        )

        self.cfg = cfg
        run_cfg = cfg["run"]
        self.run_name = run_cfg["name"]
        self.run_dir = Path(run_cfg.get("output_dir", "outputs/part3_mup")) / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.drive_run_dir = None
        drive_dir = _expand_colab_path(run_cfg.get("drive_output_dir"))
        if drive_dir is not None:
            self.drive_run_dir = drive_dir / self.run_name

        seed = int(cfg.get("seed", 42))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        device_name = run_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
        self.precision = str(run_cfg.get("precision", "bf16"))
        self.autocast_dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        self.use_autocast = self.device.type == "cuda" and self.precision in {"bf16", "fp16"}
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda" and self.precision == "fp16")

        model_cfg = cfg["model"]
        self.model_config = MupTransformerConfig(**model_cfg)
        self.model = MupDecoderOnlyTransformer(self.model_config)
        base_cfg = make_mup_base_config(self.model_config)
        delta_cfg = make_mup_base_config(self.model_config, d_model=base_cfg.d_model * 2)
        base_model = MupDecoderOnlyTransformer(base_cfg)
        delta_model = MupDecoderOnlyTransformer(delta_cfg)
        set_base_shapes(self.model, base_model, delta=delta_model)
        self.model.to(self.device)

        train_cfg = cfg["training"]
        self.optimizer = MuAdamW(
            self.model.parameters(),
            lr=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg.get("weight_decay", 0.1)),
            betas=tuple(train_cfg.get("betas", [0.9, 0.95])),
            eps=float(train_cfg.get("eps", 1e-8)),
        )

        self.train_data = TokenizedSvgDataset(cfg["data"], "train")
        self.val_data = TokenizedSvgDataset(cfg["data"], "validation")

        total_steps = self._estimate_total_steps()
        warmup_steps = int(train_cfg.get("warmup_steps", 0))
        if warmup_steps <= 0:
            warmup_steps = int(total_steps * float(train_cfg.get("warmup_ratio", 0.02)))
        self.scheduler = build_cosine_warmup_scheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=float(train_cfg.get("min_lr_ratio", 0.1)),
        )

        self.global_step = 0
        self.tokens_seen = 0
        self.next_row_index = 0
        self.best_val_loss = float("inf")
        self.started_at = time.time()
