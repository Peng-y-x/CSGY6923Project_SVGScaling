from __future__ import annotations

import argparse
import copy
import csv
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
from src.train.trainer import Part3MupTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablate muP attention scale on small models.")
    parser.add_argument("--config", default="configs/mup_attention_scale_ablation.yaml")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--best-lr-json", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--analysis-dir", default=None)
    parser.add_argument("--no-drive-sync", action="store_true")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_learning_rate(args: argparse.Namespace, cfg: dict[str, Any]) -> float:
    if args.learning_rate is not None:
        return float(args.learning_rate)
    if args.best_lr_json:
        with Path(args.best_lr_json).open("r", encoding="utf-8") as f:
            return float(json.load(f)["learning_rate"])
    return float(cfg["learning_rate"])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "run_name",
        "source_config",
        "attention_scale",
        "learning_rate",
        "num_parameters",
        "num_parameters_non_embedding",
        "val_loss",
        "val_ppl",
        "tokens_seen",
        "wall_clock_seconds",
        "peak_gpu_memory_gb",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def plot_results(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    scale_order = ["mup", "sp"]
    sizes = []
    for row in rows:
        size = str(row["source_config"]).split("/")[-1].replace("mup_", "").replace(".yaml", "")
        if size not in sizes:
            sizes.append(size)

    x = range(len(sizes))
    width = 0.36
    by_size_scale = {
        (
            str(row["source_config"]).split("/")[-1].replace("mup_", "").replace(".yaml", ""),
            row["attention_scale"],
        ): float(row["val_loss"])
        for row in rows
    }

    plt.figure(figsize=(7, 4.5))
    for idx, scale in enumerate(scale_order):
        offsets = [i + (idx - 0.5) * width for i in x]
        vals = [by_size_scale.get((size, scale), float("nan")) for size in sizes]
        label = "muP scale: 1/head_dim" if scale == "mup" else "SP scale: 1/sqrt(head_dim)"
        plt.bar(offsets, vals, width=width, label=label)
    plt.xticks(list(x), sizes)
    plt.ylabel("Validation loss after 1 epoch")
    plt.title("muP Attention Scale Ablation")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    sweep_cfg = load_yaml(args.config)
    learning_rate = resolve_learning_rate(args, sweep_cfg)
    output_dir = Path(args.output_dir or sweep_cfg.get("output_dir", "outputs/part3_attention_scale_ablation"))
    analysis_dir = Path(args.analysis_dir or sweep_cfg.get("analysis_dir", "outputs/part3_attention_scale_ablation_analysis"))
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for config_path in sweep_cfg["base_configs"]:
        base_cfg = load_yaml(config_path)
        base_name = str(base_cfg["run"]["name"])
        for attention_scale in sweep_cfg.get("attention_scales", ["mup", "sp"]):
            cfg = copy.deepcopy(base_cfg)
            cfg["model"]["attention_scale"] = attention_scale
            cfg["training"]["learning_rate"] = learning_rate
            cfg["run"]["name"] = f"{base_name}_attn_{attention_scale}"
            cfg["run"]["output_dir"] = str(output_dir)
            if not args.no_drive_sync and sweep_cfg.get("drive_output_dir"):
                cfg["run"]["drive_output_dir"] = str(sweep_cfg["drive_output_dir"])
            else:
                cfg["run"].pop("drive_output_dir", None)
            cfg["run"]["resume"] = args.resume if args.resume is not None else sweep_cfg.get("resume", "auto")
            if "max_train_tokens" in sweep_cfg:
                cfg["training"]["max_train_tokens"] = int(sweep_cfg["max_train_tokens"])
            if "max_eval_tokens" in sweep_cfg:
                cfg["training"]["max_eval_tokens"] = int(sweep_cfg["max_eval_tokens"])

            print(
                f"[attention-scale-ablation] config={config_path} scale={attention_scale} "
                f"lr={learning_rate} run={cfg['run']['name']}",
                flush=True,
            )
            trainer = Part3MupTrainer(cfg)
            metrics = trainer.train_one_epoch()
            row = {
                "source_config": config_path,
                "attention_scale": attention_scale,
                "learning_rate": learning_rate,
                **metrics,
            }
            rows.append(row)
            write_json(analysis_dir / "attention_scale_ablation.json", {"runs": rows})
            write_csv(analysis_dir / "attention_scale_ablation.csv", rows)

    plot_results(analysis_dir / "attention_scale_ablation.png", rows)
    if not args.no_drive_sync and sweep_cfg.get("drive_analysis_dir"):
        shutil.copytree(analysis_dir, Path(sweep_cfg["drive_analysis_dir"]), dirs_exist_ok=True)
    print(json.dumps({"runs": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
