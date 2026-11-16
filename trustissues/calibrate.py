"""Fit a sensor calibration line you can actually trust.

    python -m trustissues.calibrate readings.csv            # columns: raw,truth
    python -m trustissues.calibrate readings.csv --plot fit.png

Fits RANSAC, then refits ordinary least squares on the inliers only (RANSAC
picks *who* to believe; least squares is still the best judge of the honest
ones). Prints the line, the readings it refused to believe, and what plain
least squares would have said, so you can see what the liars would have cost you.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor


@dataclass
class Fit:
    slope: float
    offset: float
    inliers: np.ndarray       # bool mask over the input rows
    ols_slope: float
    ols_offset: float


def robust_line(raw: np.ndarray, truth: np.ndarray, threshold: float | None = None) -> Fit:
    X = raw.reshape(-1, 1)
    if threshold is None:
        # Scale-free default: 2.5x the median absolute deviation of a first rough fit.
        rough = np.polyfit(raw, truth, 1)
        resid = truth - np.polyval(rough, raw)
        threshold = 2.5 * 1.4826 * np.median(np.abs(resid - np.median(resid))) or 1e-9
    r = RANSACRegressor(residual_threshold=threshold, random_state=0).fit(X, truth)
    keep = r.inlier_mask_
    final = LinearRegression().fit(X[keep], truth[keep])
    ols = LinearRegression().fit(X, truth)
    return Fit(float(final.coef_[0]), float(final.intercept_), keep,
               float(ols.coef_[0]), float(ols.intercept_))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv")
    ap.add_argument("--threshold", type=float, help="inlier band in truth units (default: from the data)")
    ap.add_argument("--plot")
    a = ap.parse_args()

    with open(a.csv) as f:
        rows = [(float(r["raw"]), float(r["truth"])) for r in csv.DictReader(f)]
    raw, truth = map(np.array, zip(*rows))
    fit = robust_line(raw, truth, a.threshold)

    print(f"truth = {fit.slope:.6g} * raw + {fit.offset:.6g}")
    print(f"believed {fit.inliers.sum()} of {len(raw)} readings")
    for i in np.where(~fit.inliers)[0]:
        print(f"  refused row {i + 2}: raw={raw[i]:g} truth={truth[i]:g}")
    span = np.linspace(raw.min(), raw.max(), 50)
    cost = np.max(np.abs((fit.ols_slope - fit.slope) * span + fit.ols_offset - fit.offset))
    print(f"plain least squares would be off by up to {cost:.4g} truth units across this range")

    if a.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
        ax.scatter(raw[fit.inliers], truth[fit.inliers], s=10, color="#333", label="believed")
        ax.scatter(raw[~fit.inliers], truth[~fit.inliers], s=14, marker="x", color="#b3261e", label="refused")
        ax.plot(span, fit.slope * span + fit.offset, color="#1f4e8c", lw=2, label="calibration")
        ax.plot(span, fit.ols_slope * span + fit.ols_offset, color="#8a8a8a", lw=1.2, ls="--", label="what OLS would say")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(a.plot)


if __name__ == "__main__":
    main()
