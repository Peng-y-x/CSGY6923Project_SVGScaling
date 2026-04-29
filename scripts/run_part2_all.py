from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_CONFIGS = [
    "configs/train_tiny.yaml",
    "configs/train_small.yaml",
    "configs/train_medium.yaml",
    "configs/train_large.yaml",
    "configs/train_xl.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all five Part 2 model sizes.")
    parser.add_argument("--best-lr-json", default="outputs/part2_lr_sweep/best_lr.json")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lr = args.learning_rate
    if lr is None:
        best_path = Path(args.best_lr_json)
        if not best_path.exists():
            raise FileNotFoundError(
                f"Missing {best_path}. Run scripts/run_lr_sweep.py first or pass --learning-rate."
            )
        with best_path.open("r", encoding="utf-8") as f:
            lr = float(json.load(f)["learning_rate"])

    for cfg in args.configs:
        cmd = [sys.executable, "scripts/run_train.py", "--config", cfg, "--learning-rate", str(lr)]
        print("[part2]", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
