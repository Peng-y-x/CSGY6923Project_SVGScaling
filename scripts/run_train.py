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

from src.train.trainer import Part2Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one Part 2 standard-parameterization model.")
    parser.add_argument("--config", required=True, help="Path to train yaml.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override config LR.")
    parser.add_argument("--run-name", type=str, default=None, help="Override run.name.")
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
    if args.resume is not None:
        cfg["run"]["resume"] = args.resume

    trainer = Part2Trainer(cfg)
    metrics = trainer.train_one_epoch()
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
