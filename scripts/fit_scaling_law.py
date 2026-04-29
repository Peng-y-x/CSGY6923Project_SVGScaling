from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.plots import plot_scaling_curve
from src.analysis.scaling_fit import fit_power_law
from src.train.metrics import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit Part 2 scaling law from final_metrics.json files.")
    parser.add_argument("--runs-dir", default="outputs/part2", help="Directory containing per-run folders.")
    parser.add_argument("--output-dir", default="outputs/part2_analysis", help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(runs_dir.glob("*/final_metrics.json")):
        with path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        rows.append(metrics)

    if len(rows) < 5:
        raise RuntimeError(f"Expected at least 5 model runs for Part 2; found {len(rows)} in {runs_dir}")

    rows.sort(key=lambda x: int(x["num_parameters"]))
    fit = fit_power_law([int(x["num_parameters"]) for x in rows], [float(x["val_loss"]) for x in rows])

    csv_path = output_dir / "part2_scaling_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_name",
                "num_parameters",
                "val_loss",
                "val_ppl",
                "tokens_seen",
                "wall_clock_seconds",
                "tokens_per_second_epoch",
                "peak_gpu_memory_gb",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in writer.fieldnames})

    write_json(
        output_dir / "part2_scaling_fit.json",
        {
            "fit": fit.__dict__,
            "runs": rows,
            "formula": "L = a * N^(-alpha) + c",
        },
    )
    plot_scaling_curve(
        param_counts=[int(x["num_parameters"]) for x in rows],
        losses=[float(x["val_loss"]) for x in rows],
        labels=[str(x["run_name"]) for x in rows],
        fit=fit,
        output_path=output_dir / "part2_scaling_curve.png",
    )
    print(json.dumps({"alpha": fit.alpha, "a": fit.a, "c": fit.c, "table": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()
