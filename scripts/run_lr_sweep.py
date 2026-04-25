from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train.metrics import write_json
from src.train.trainer import Part2Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 2 LR sweep on the smallest model.")
    parser.add_argument("--config", default="configs/sweep_lr.yaml", help="Path to sweep yaml.")
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    sweep_cfg = load_config(args.config)
    base_cfg = load_config(sweep_cfg["base_config"])
    learning_rates = [float(x) for x in sweep_cfg["learning_rates"]]
    output_dir = Path(sweep_cfg.get("output_dir", "outputs/part2_lr_sweep"))
    output_dir.mkdir(parents=True, exist_ok=True)
    drive_output_dir = sweep_cfg.get("drive_output_dir")

    rows: list[dict[str, Any]] = []
    for lr in learning_rates:
        cfg = copy.deepcopy(base_cfg)
        cfg["training"]["learning_rate"] = lr
        cfg["run"]["name"] = f"{sweep_cfg.get('run_prefix', 'tiny_lr')}_{lr:.1e}".replace("+", "")
        cfg["run"]["output_dir"] = str(output_dir)
        cfg["run"]["resume"] = sweep_cfg.get("resume", "auto")
        if "max_train_tokens" in sweep_cfg:
            cfg["training"]["max_train_tokens"] = int(sweep_cfg["max_train_tokens"])
        if "max_eval_tokens" in sweep_cfg:
            cfg["training"]["max_eval_tokens"] = int(sweep_cfg["max_eval_tokens"])

        print(f"[sweep] lr={lr} run={cfg['run']['name']}")
        trainer = Part2Trainer(cfg)
        metrics = trainer.train_one_epoch()
        row = {"learning_rate": lr, **metrics}
        rows.append(row)
        write_json(output_dir / "sweep_results.json", {"runs": rows})
        if drive_output_dir:
            shutil.copytree(output_dir, Path(drive_output_dir), dirs_exist_ok=True)

    best = min(rows, key=lambda x: x["val_loss"])
    write_json(output_dir / "best_lr.json", best)
    if drive_output_dir:
        shutil.copytree(output_dir, Path(drive_output_dir), dirs_exist_ok=True)
    print("[sweep] best:")
    print(json.dumps(best, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
