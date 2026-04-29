from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


@dataclass
class ScalingFit:
    a: float
    alpha: float
    c: float
    covariance: list[list[float]]


def power_law(params: np.ndarray | float, a: float, alpha: float, c: float) -> np.ndarray:
    n = np.asarray(params, dtype=float)
    return a * np.power(n, -alpha) + c


def fit_power_law(param_counts: list[int], losses: list[float]) -> ScalingFit:
    x = np.asarray(param_counts, dtype=float)
    y = np.asarray(losses, dtype=float)
    if len(x) < 3:
        raise ValueError("Need at least three points to fit L = a * N^-alpha + c.")

    c0 = max(0.0, float(y.min()) - 0.2)
    p0 = [float((y.max() - c0) * (x.min() ** 0.1)), 0.1, c0]
    bounds = ([0.0, 0.0, 0.0], [np.inf, 5.0, float(y.min())])
    popt, pcov = curve_fit(power_law, x, y, p0=p0, bounds=bounds, maxfev=50000)
    return ScalingFit(a=float(popt[0]), alpha=float(popt[1]), c=float(popt[2]), covariance=pcov.tolist())
