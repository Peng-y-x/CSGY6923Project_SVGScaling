from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.svg_metrics import evaluate_svg_string
from src.generation.render import render_svg_to_png
from src.models.checkpoint import load_checkpoint_model
from src.train.trainer import TokenizedSvgDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Part 4 model and generated SVG samples.")
    parser.add_argument("--train-config", default="configs/part4_best.yaml")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--samples-jsonl", default="outputs/part4_samples/samples.jsonl")
    parser.add_argument("--output-dir", default="outputs/part4_eval")
    parser.add_argument("--drive-output-dir", default=None)
    parser.add_argument("--max-test-tokens", type=int, default=0)
    return parser.parse_args()


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_test_perplexity(
    cfg: dict[str, Any],
    model: torch.nn.Module,
    *,
    max_test_tokens: int | None,
    device: torch.device,
) -> dict[str, float]:
    data = TokenizedSvgDataset(cfg["data"], "test")
    train_cfg = cfg["training"]
    losses: list[float] = []
    total_tokens = 0
    precision = str(cfg.get("run", {}).get("precision", "bf16"))
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    use_autocast = device.type == "cuda" and precision in {"bf16", "fp16"}
    model.eval()
    with torch.no_grad():
        for _, x, y, mask, ntok in data.iter_batches(
            start_index=0,
            tokens_per_batch=int(train_cfg["tokens_per_batch"]),
            max_padded_tokens_per_batch=int(train_cfg.get("max_padded_tokens_per_batch", 0)) or None,
            block_size=int(model.cfg.block_size),
            pad_token_id=int(model.cfg.pad_token_id),
            max_tokens=max_test_tokens,
        ):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                _, loss = model(x, labels=y, attention_mask=mask)
            assert loss is not None
            losses.append(float(loss.item()) * ntok)
            total_tokens += ntok
    test_loss = sum(losses) / max(1, total_tokens)
    return {
        "test_loss": test_loss,
        "test_perplexity": math.exp(min(20.0, test_loss)),
        "test_tokens": float(total_tokens),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.train_config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device_name = str(cfg.get("run", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    model, _ = load_checkpoint_model(args.checkpoint_path, device)

    ppl_metrics = evaluate_test_perplexity(
        cfg,
        model,
        max_test_tokens=int(args.max_test_tokens) or None,
        device=device,
    )

    sample_rows = _read_jsonl(Path(args.samples_jsonl))
    detail_rows = []
    render_dir = output_dir / "render_check"
    for row in sample_rows:
        svg = row.get("svg", "")
        png_path = render_dir / f"{row.get('id', len(detail_rows))}.png"
        render_ok = render_svg_to_png(svg, png_path)
        validity = evaluate_svg_string(svg, render_valid=render_ok)
        detail = {
            "id": row.get("id", ""),
            "kind": row.get("kind", ""),
            "temperature": row.get("temperature"),
            "xml_valid": validity.xml_valid,
            "root_is_svg": validity.root_is_svg,
            "structural_valid": validity.structural_valid,
            "render_valid": validity.render_valid,
            "error": validity.error,
        }
        detail_rows.append(detail)

    n = max(1, len(detail_rows))
    gen_metrics = {
        "num_generated": len(detail_rows),
        "xml_validity_rate": sum(1 for r in detail_rows if r["xml_valid"]) / n,
        "svg_render_rate": sum(1 for r in detail_rows if r["render_valid"]) / n,
        "structural_validity_rate": sum(1 for r in detail_rows if r["structural_valid"]) / n,
        "root_svg_rate": sum(1 for r in detail_rows if r["root_is_svg"]) / n,
    }
    metrics = {**ppl_metrics, **gen_metrics}

    (output_dir / "part4_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "sample_validity.jsonl").open("w", encoding="utf-8") as f:
        for row in detail_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.drive_output_dir:
        drive_path = Path(os.path.expandvars(os.path.expanduser(args.drive_output_dir)))
        drive_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(output_dir, drive_path, dirs_exist_ok=True)
        print(f"[eval] Synced results to Drive: {drive_path}")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
