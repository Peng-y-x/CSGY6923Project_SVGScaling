from __future__ import annotations

from pathlib import Path

import numpy as np

from src.analysis.scaling_fit import ScalingFit, power_law


def plot_scaling_curve(
    *,
    param_counts: list[int],
    losses: list[float],
    labels: list[str],
    fit: ScalingFit,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(param_counts, dtype=float)
    y = np.asarray(losses, dtype=float)
    xs = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, s=70, label="runs")
    for xi, yi, label in zip(x, y, labels):
        plt.annotate(label, (xi, yi), textcoords="offset points", xytext=(5, 5))
    plt.plot(xs, power_law(xs, fit.a, fit.alpha, fit.c), label=f"fit alpha={fit.alpha:.3f}")
    plt.xscale("log")
    plt.xlabel("Number of parameters")
    plt.ylabel("Validation loss after 1 epoch")
    plt.title("Part 2 SVG Transformer Scaling")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
