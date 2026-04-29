from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.scaling_fit import ScalingFit, fit_power_law, power_law
from src.train.metrics import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Part 2 SP scaling vs Part 3 muP and extrapolate 10x.")
    parser.add_argument("--sp-runs-dir", default="outputs/part2", help="Standard-parameterization runs.")
    parser.add_argument("--fallback-sp-runs-dir", default=None, help="Fallback SP runs if primary is absent.")
    parser.add_argument("--mup-runs-dir", default="outputs/part3_mup", help="muP runs.")
    parser.add_argument("--output-dir", default="outputs/part3_analysis", help="Analysis output directory.")
    parser.add_argument("--drive-output-dir", default=None, help="Optional Drive directory to sync outputs.")
    parser.add_argument("--sp-sweep-json", default="outputs/part2_lr_sweep/sweep_results.json")
    parser.add_argument("--mup-sweep-json", default="outputs/part3_mup_lr_sweep/sweep_results.json")
    parser.add_argument("--allow-partial", action="store_true", help="Allow analysis with fewer than five size runs.")
    return parser.parse_args()


def load_runs(runs_dir: Path, expected_names: set[str]) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(runs_dir.glob("*/final_metrics.json")):
        with path.open("r", encoding="utf-8") as f:
            row = json.load(f)
        if str(row.get("run_name", path.parent.name)) not in expected_names:
            continue
        row["_run_dir"] = str(path.parent)
        rows.append(row)
    rows.sort(key=lambda x: int(x["num_parameters"]))
    return rows


def require_expected(rows: list[dict[str, Any]], expected_names: set[str], label: str, allow_partial: bool) -> None:
    found = {str(row.get("run_name")) for row in rows}
    missing = sorted(expected_names - found)
    if missing and not allow_partial:
        raise RuntimeError(
            f"{label} is incomplete. Missing final_metrics.json for: {missing}. "
            "Do not use the Part 3 scaling fit/extrapolation until all five size runs finish."
        )
    if len(rows) < 3:
        raise RuntimeError(f"Need at least 3 {label} runs to fit a power law; found {len(rows)}.")


def fit_with_stats(rows: list[dict[str, Any]]) -> tuple[ScalingFit, float]:
    fit = fit_power_law([int(x["num_parameters"]) for x in rows], [float(x["val_loss"]) for x in rows])
    x = np.asarray([int(r["num_parameters"]) for r in rows], dtype=float)
    y = np.asarray([float(r["val_loss"]) for r in rows], dtype=float)
    pred = power_law(x, fit.a, fit.alpha, fit.c)
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    return fit, rmse


def prediction_interval(fit: ScalingFit, rmse: float, n_params: float) -> dict[str, float]:
    pred = float(power_law(n_params, fit.a, fit.alpha, fit.c))
    try:
        cov = np.asarray(fit.covariance, dtype=float)
        grad = np.asarray(
            [
                n_params ** (-fit.alpha),
                -fit.a * math.log(n_params) * (n_params ** (-fit.alpha)),
                1.0,
            ],
            dtype=float,
        )
        fit_var = float(grad @ cov @ grad.T)
        sigma = math.sqrt(max(0.0, fit_var) + rmse**2)
    except Exception:
        sigma = rmse
    return {
        "num_parameters": float(n_params),
        "predicted_val_loss": pred,
        "uncertainty_sigma": float(sigma),
        "ci95_low": float(pred - 1.96 * sigma),
        "ci95_high": float(pred + 1.96 * sigma),
    }


def write_table(path: Path, rows: list[dict[str, Any]], parameterization: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "parameterization",
                "run_name",
                "num_parameters",
                "num_parameters_non_embedding",
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
            out = {k: row.get(k, "") for k in writer.fieldnames}
            out["parameterization"] = parameterization
            writer.writerow(out)


def plot_comparison(
    *,
    sp_rows: list[dict[str, Any]],
    mup_rows: list[dict[str, Any]],
    sp_fit: ScalingFit,
    mup_fit: ScalingFit,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    sp_x = np.asarray([int(r["num_parameters"]) for r in sp_rows], dtype=float)
    sp_y = np.asarray([float(r["val_loss"]) for r in sp_rows], dtype=float)
    mup_x = np.asarray([int(r["num_parameters"]) for r in mup_rows], dtype=float)
    mup_y = np.asarray([float(r["val_loss"]) for r in mup_rows], dtype=float)
    all_x = np.concatenate([sp_x, mup_x])
    xs = np.logspace(np.log10(all_x.min()), np.log10(all_x.max()), 300)

    plt.figure(figsize=(8, 5.5))
    plt.scatter(sp_x, sp_y, s=70, marker="o", label="SP runs")
    plt.scatter(mup_x, mup_y, s=70, marker="s", label="muP runs")
    plt.plot(xs, power_law(xs, sp_fit.a, sp_fit.alpha, sp_fit.c), label=f"SP fit alpha={sp_fit.alpha:.3f}")
    plt.plot(xs, power_law(xs, mup_fit.a, mup_fit.alpha, mup_fit.c), label=f"muP fit alpha={mup_fit.alpha:.3f}")
    for row in sp_rows:
        plt.annotate(str(row["run_name"]), (int(row["num_parameters"]), float(row["val_loss"])), xytext=(5, 5), textcoords="offset points", fontsize=8)
    for row in mup_rows:
        plt.annotate(str(row["run_name"]), (int(row["num_parameters"]), float(row["val_loss"])), xytext=(5, -12), textcoords="offset points", fontsize=8)
    plt.xscale("log")
    plt.xlabel("Number of parameters")
    plt.ylabel("Validation loss after 1 epoch")
    plt.title("Part 3 Standard Parameterization vs muP Scaling")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def load_sweep(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload.get("runs", payload if isinstance(payload, list) else [])
    return sorted(rows, key=lambda x: float(x["learning_rate"]))


def write_sweep_table(path: Path, sp_rows: list[dict[str, Any]], mup_rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["parameterization", "learning_rate", "val_loss", "val_ppl", "run_name"],
        )
        writer.writeheader()
        for parameterization, rows in [("sp", sp_rows), ("mup", mup_rows)]:
            for row in rows:
                writer.writerow(
                    {
                        "parameterization": parameterization,
                        "learning_rate": row.get("learning_rate"),
                        "val_loss": row.get("val_loss"),
                        "val_ppl": row.get("val_ppl"),
                        "run_name": row.get("run_name"),
                    }
                )


def plot_lr_sweeps(
    *,
    sp_rows: list[dict[str, Any]],
    mup_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    if not sp_rows and not mup_rows:
        return
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4.8))
    if sp_rows:
        plt.plot(
            [float(r["learning_rate"]) for r in sp_rows],
            [float(r["val_loss"]) for r in sp_rows],
            marker="o",
            label="SP Tiny LR sweep",
        )
    if mup_rows:
        plt.plot(
            [float(r["learning_rate"]) for r in mup_rows],
            [float(r["val_loss"]) for r in mup_rows],
            marker="s",
            label="muP Tiny LR sweep",
        )
    plt.xscale("log")
    plt.xlabel("Learning rate")
    plt.ylabel("Validation loss after 1 epoch")
    plt.title("Tiny Model Learning Rate Sweep")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sp_runs_dir = Path(args.sp_runs_dir)
    if not list(sp_runs_dir.glob("*/final_metrics.json")) and args.fallback_sp_runs_dir:
        sp_runs_dir = Path(args.fallback_sp_runs_dir)
    mup_runs_dir = Path(args.mup_runs_dir)

    sp_expected = {"tiny", "small", "medium", "large", "xl"}
    mup_expected = {"tiny_mup", "small_mup", "medium_mup", "large_mup", "xl_mup"}
    sp_rows = load_runs(sp_runs_dir, sp_expected)
    mup_rows = load_runs(mup_runs_dir, mup_expected)
    require_expected(sp_rows, sp_expected, "SP baseline", args.allow_partial)
    require_expected(mup_rows, mup_expected, "muP runs", args.allow_partial)

    sp_fit, sp_rmse = fit_with_stats(sp_rows)
    mup_fit, mup_rmse = fit_with_stats(mup_rows)
    best_name, best_fit, best_rmse, best_rows = (
        ("mup", mup_fit, mup_rmse, mup_rows) if mup_rmse <= sp_rmse else ("sp", sp_fit, sp_rmse, sp_rows)
    )
    largest_params = max(int(r["num_parameters"]) for r in best_rows)
    extrapolation = prediction_interval(best_fit, best_rmse, largest_params * 10.0)

    write_table(output_dir / "sp_scaling_table.csv", sp_rows, "sp")
    write_table(output_dir / "mup_scaling_table.csv", mup_rows, "mup")
    sp_sweep = load_sweep(Path(args.sp_sweep_json))
    mup_sweep = load_sweep(Path(args.mup_sweep_json))
    write_sweep_table(output_dir / "lr_sweep_comparison.csv", sp_sweep, mup_sweep)
    plot_comparison(
        sp_rows=sp_rows,
        mup_rows=mup_rows,
        sp_fit=sp_fit,
        mup_fit=mup_fit,
        output_path=output_dir / "part3_sp_vs_mup_scaling.png",
    )
    plot_lr_sweeps(
        sp_rows=sp_sweep,
        mup_rows=mup_sweep,
        output_path=output_dir / "part3_sp_vs_mup_lr_sweep.png",
    )

    result = {
        "sp_runs_dir": str(sp_runs_dir),
        "mup_runs_dir": str(mup_runs_dir),
        "sp_fit": {**sp_fit.__dict__, "rmse": sp_rmse},
        "mup_fit": {**mup_fit.__dict__, "rmse": mup_rmse},
        "best_fit_used_for_extrapolation": best_name,
        "extrapolation_10x_largest": extrapolation,
        "sweep_rows": {"sp": len(sp_sweep), "mup": len(mup_sweep)},
        "formula": "L = a * N^(-alpha) + c",
        "notes": "The interval combines curve_fit covariance with in-sample RMSE; it is a rough uncertainty estimate, not a held-out guarantee.",
    }
    write_json(output_dir / "part3_scaling_comparison.json", result)

    if args.drive_output_dir:
        shutil.copytree(output_dir, Path(args.drive_output_dir), dirs_exist_ok=True)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
