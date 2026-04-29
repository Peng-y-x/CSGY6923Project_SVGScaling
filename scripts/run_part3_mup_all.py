from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_CONFIGS = [
    "configs/mup_tiny.yaml",
    "configs/mup_small.yaml",
    "configs/mup_medium.yaml",
    "configs/mup_large.yaml",
    "configs/mup_xl.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all five Part 3 muP model sizes.")
    parser.add_argument("--best-lr-json", default="outputs/part3_mup_lr_sweep/best_lr.json")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--drive-output-dir", default=None)
    parser.add_argument("--run-suffix", default=None)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lr = args.learning_rate
    if lr is None:
        best_path = Path(args.best_lr_json)
        if not best_path.exists():
            raise FileNotFoundError(
                f"Missing {best_path}. Run scripts/run_mup_lr_sweep.py first or pass --learning-rate."
            )
        with best_path.open("r", encoding="utf-8") as f:
            lr = float(json.load(f)["learning_rate"])

    for cfg in args.configs:
        cmd = [
            sys.executable,
            "scripts/run_mup_train.py",
            "--config",
            cfg,
            "--learning-rate",
            str(lr),
        ]
        if args.output_dir:
            cmd.extend(["--output-dir", args.output_dir])
        if args.drive_output_dir:
            cmd.extend(["--drive-output-dir", args.drive_output_dir])
        if args.run_suffix:
            cmd.extend(["--run-suffix", args.run_suffix])
        if args.resume is not None:
            cmd.extend(["--resume", args.resume])
        print("[part3-mup]", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
