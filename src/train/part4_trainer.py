from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from src.train.metrics import append_jsonl, write_json
from src.train.trainer import Part2Trainer, Part3MupTrainer


class Part4TrainingMixin:
    completed_epochs: int

    def _estimate_total_steps(self) -> int:
        one_epoch_steps = super()._estimate_total_steps()  # type: ignore[misc]
        max_epochs = int(self.cfg.get("training", {}).get("max_epochs", 1))
        return max(1, one_epoch_steps * max(1, max_epochs))

    def _maybe_init_from_checkpoint(self) -> None:
        init_from = self.cfg["run"].get("init_from_checkpoint")
        if not init_from:
            return

        local_latest = self._latest_checkpoint_path(self.run_dir)
        drive_latest = self._latest_checkpoint_path(self.drive_run_dir) if self.drive_run_dir is not None else None
        if local_latest.exists() or (drive_latest is not None and drive_latest.exists()):
            return

        init_path = Path(init_from)
        if not init_path.exists():
            raise FileNotFoundError(f"init_from_checkpoint does not exist: {init_path}")
        ckpt = torch.load(init_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.next_row_index = 0
        self.tokens_seen = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        print(f"[init] Loaded model weights from: {init_path}")

    def _checkpoint_payload(self) -> dict[str, Any]:
        payload = super()._checkpoint_payload()  # type: ignore[misc]
        payload["completed_epochs"] = int(getattr(self, "completed_epochs", 0))
        return payload

    def maybe_resume(self) -> None:
        super().maybe_resume()  # type: ignore[misc]

        candidates: list[Path] = []
        explicit = self.cfg["run"].get("resume_from")
        if explicit:
            candidates.append(Path(explicit))
        candidates.append(self._latest_checkpoint_path(self.run_dir))
        if self.drive_run_dir is not None:
            candidates.append(self._latest_checkpoint_path(self.drive_run_dir))

        ckpt_path = next((p for p in candidates if p.exists()), None)
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.completed_epochs = int(ckpt.get("completed_epochs", 0))

    def train_epochs(self) -> dict[str, Any]:
        self.completed_epochs = int(getattr(self, "completed_epochs", 0))
        self._maybe_init_from_checkpoint()
        self.maybe_resume()
        self.model.train()

        train_cfg = self.cfg["training"]
        max_epochs = int(train_cfg.get("max_epochs", 1))
        if max_epochs <= 0:
            raise ValueError("training.max_epochs must be positive")

        log_interval = int(train_cfg.get("log_interval_steps", 20))
        eval_interval = int(train_cfg.get("eval_interval_steps", 0))
        checkpoint_interval = int(train_cfg.get("checkpoint_interval_steps", 200))
        max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
        max_train_tokens_per_epoch = int(train_cfg.get("max_train_tokens_per_epoch", 0)) or None
        metrics_path = self.run_dir / "metrics.jsonl"

        while self.completed_epochs < max_epochs:
            if self.next_row_index >= len(self.train_data):
                self.next_row_index = 0

            epoch_index = self.completed_epochs + 1
            print(f"[part4] Starting epoch {epoch_index}/{max_epochs} at row {self.next_row_index}")
            step_start = time.time()
            recent_loss = 0.0
            recent_tokens = 0

            for next_row, x, y, mask, ntok in self.train_data.iter_batches(
                start_index=self.next_row_index,
                tokens_per_batch=int(train_cfg["tokens_per_batch"]),
                max_padded_tokens_per_batch=int(train_cfg.get("max_padded_tokens_per_batch", 0)) or None,
                block_size=int(self.model_config.block_size),
                pad_token_id=int(self.model_config.pad_token_id),
                max_tokens=max_train_tokens_per_epoch,
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
                        "epoch": epoch_index,
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
                    metrics.update(
                        {
                            "type": "eval",
                            "epoch": epoch_index,
                            "step": self.global_step,
                            "tokens_seen": self.tokens_seen,
                        }
                    )
                    append_jsonl(metrics_path, metrics)
                    is_best = metrics["val_loss"] < self.best_val_loss
                    if is_best:
                        self.best_val_loss = metrics["val_loss"]
                    self.save_checkpoint(is_best=is_best)

                if checkpoint_interval > 0 and self.global_step % checkpoint_interval == 0:
                    self.save_checkpoint()

            self.completed_epochs += 1
            self.next_row_index = 0
            epoch_metrics = self.evaluate()
            epoch_metrics.update(
                {
                    "type": "epoch_end",
                    "epoch": self.completed_epochs,
                    "step": self.global_step,
                    "tokens_seen": self.tokens_seen,
                }
            )
            append_jsonl(metrics_path, epoch_metrics)
            is_best = epoch_metrics["val_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = epoch_metrics["val_loss"]
            self.save_checkpoint(is_best=is_best)

        final_metrics = self.evaluate()
        final_metrics.update(self._final_summary_fields())
        final_metrics["completed_epochs"] = self.completed_epochs
        write_json(self.run_dir / "final_metrics.json", final_metrics)
        append_jsonl(metrics_path, {"type": "final", **final_metrics})
        self.save_checkpoint(is_best=final_metrics["val_loss"] <= self.best_val_loss)
        return final_metrics


class Part4Trainer(Part4TrainingMixin, Part2Trainer):
    pass


class Part4MupTrainer(Part4TrainingMixin, Part3MupTrainer):
    pass
