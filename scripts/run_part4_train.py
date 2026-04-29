from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train.part4_trainer import Part4MupTrainer, Part4Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Part 4 best model for extra epochs/tokens.")
    parser.add_argument("--config", required=True, help="Path to Part 4 train yaml.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override config LR.")
    parser.add_argument("--best-lr-json", default=None, help="Read learning_rate from a Part 2/3 best_lr.json.")
    parser.add_argument("--parameterization", choices=["auto", "sp", "mup"], default="auto")
    parser.add_argument("--init-from", default=None, help="Initialize model weights from an earlier checkpoint.")
    parser.add_argument("--run-name", default=None, help="Override run.name.")
    parser.add_argument("--output-dir", default=None, help="Override run.output_dir.")
    parser.add_argument("--drive-output-dir", default=None, help="Override run.drive_output_dir.")
    parser.add_argument("--resume", default=None, help="Override run.resume.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override training.max_epochs.")
    parser.add_argument(
        "--max-train-tokens-per-epoch",
        type=int,
        default=None,
        help="Optional cap for faster smoke tests or limited Colab budgets.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    lr = args.learning_rate
    if lr is None and args.best_lr_json:
        with Path(args.best_lr_json).open("r", encoding="utf-8") as f:
            lr = float(json.load(f)["learning_rate"])
    if lr is not None:
        cfg["training"]["learning_rate"] = lr

    if args.run_name:
        cfg["run"]["name"] = args.run_name
    if args.output_dir:
        cfg["run"]["output_dir"] = args.output_dir
    if args.drive_output_dir:
        cfg["run"]["drive_output_dir"] = args.drive_output_dir
    if args.resume is not None:
        cfg["run"]["resume"] = args.resume
    if args.init_from:
        cfg["run"]["init_from_checkpoint"] = args.init_from
    if args.max_epochs is not None:
        cfg["training"]["max_epochs"] = args.max_epochs
    if args.max_train_tokens_per_epoch is not None:
        cfg["training"]["max_train_tokens_per_epoch"] = args.max_train_tokens_per_epoch

    parameterization = args.parameterization
    if parameterization == "auto":
        parameterization = "mup" if "base_d_model" in cfg.get("model", {}) else "sp"

    trainer_cls = Part4MupTrainer if parameterization == "mup" else Part4Trainer
    trainer = trainer_cls(cfg)
    metrics = trainer.train_epochs()
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
