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

from src.train.trainer import Part3MupTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one Part 3 muP model.")
    parser.add_argument("--config", required=True, help="Path to muP train yaml.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override config LR.")
    parser.add_argument("--run-name", type=str, default=None, help="Override run.name.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override run.output_dir.")
    parser.add_argument("--drive-output-dir", type=str, default=None, help="Override run.drive_output_dir.")
    parser.add_argument("--run-suffix", type=str, default=None, help="Append suffix to run.name.")
    parser.add_argument("--resume", type=str, default=None, help="Override run.resume.")
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.learning_rate is not None:
        cfg["training"]["learning_rate"] = args.learning_rate
    if args.run_name:
        cfg["run"]["name"] = args.run_name
    if args.run_suffix:
        cfg["run"]["name"] = f"{cfg['run']['name']}_{args.run_suffix}"
    if args.output_dir:
        cfg["run"]["output_dir"] = args.output_dir
    if args.drive_output_dir:
        cfg["run"]["drive_output_dir"] = args.drive_output_dir
    if args.resume is not None:
        cfg["run"]["resume"] = args.resume

    trainer = Part3MupTrainer(cfg)
    metrics = trainer.train_one_epoch()
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
