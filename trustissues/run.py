"""Calibrate a depth sensor while some of its readings lie.

A pressure sensor on an underwater robot is a straight line in disguise:
depth = slope * counts + offset. You calibrate it by lowering the vehicle to
known depths and fitting a line. Bubbles, a cable tug or a bit-flip on the
I2C bus put spikes in the data. This script raises the fraction of lying
readings from 0 to 60 % and asks four estimators for the slope.

    python -m trustissues.run         # prints the table, writes assets/*.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import HuberRegressor, LinearRegression, RANSACRegressor, TheilSenRegressor

TRUE_SLOPE = 0.0125  # metres per count  (made-up sensor, honest maths)
TRUE_OFFSET = -1.8
N = 120
FRACTIONS = np.arange(0.0, 0.61, 0.05)
TRIALS = 40
ASSETS = Path(__file__).resolve().parent.parent / "assets"

ESTIMATORS = {
    "least squares (2023)": lambda: LinearRegression(),
    "Huber": lambda: HuberRegressor(epsilon=1.35, max_iter=500),
    "Theil–Sen": lambda: TheilSenRegressor(random_state=0, max_subpopulation=2000),
    "RANSAC": lambda: RANSACRegressor(residual_threshold=0.15, random_state=0),
}
COLORS = ["#8a8a8a", "#c2571a", "#2a6f4e", "#1f4e8c"]


def calibration_run(frac: float, rng: np.random.Generator):
    counts = rng.uniform(150, 1000, N)
    depth = TRUE_SLOPE * counts + TRUE_OFFSET + rng.normal(0, 0.03, N)
    bad = rng.random(N) < frac
    # Liars all lie the same way: spikes read deep (bubbles on the port, cable tugs).
    depth[bad] += rng.uniform(2.0, 6.0, bad.sum())
    return counts.reshape(-1, 1), depth, bad


GRID = np.linspace(150, 1000, 50).reshape(-1, 1)
TRUTH = TRUE_SLOPE * GRID[:, 0] + TRUE_OFFSET


def depth_error_cm(model) -> float:
    """Mean |depth error| over the sensor's working range, in centimetres."""
    return float(np.mean(np.abs(model.predict(GRID) - TRUTH)) * 100)


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    rng = np.random.default_rng(2023)
    err = {k: np.zeros((len(FRACTIONS), TRIALS)) for k in ESTIMATORS}
    for i, f in enumerate(FRACTIONS):
        for t in range(TRIALS):
            X, y, _ = calibration_run(f, rng)
            for name, make in ESTIMATORS.items():
                m = make().fit(X, y)
                err[name][i, t] = depth_error_cm(m)

    plt.rcParams.update({"font.family": "DejaVu Sans Mono", "font.size": 10})

    # 1. breakdown curves (median error over trials)
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    for (name, e), c in zip(err.items(), COLORS):
        ax.plot(FRACTIONS * 100, np.median(e, axis=1), color=c, lw=2.2, label=name)
    ax.set_yscale("log")
    ax.axhline(5, color="k", lw=0.8, alpha=0.4)
    ax.text(1, 5.6, "5 cm: a gate you still fit through", fontsize=8, alpha=0.6)
    ax.set_xlabel("readings that lie (%)")
    ax.set_ylabel("median depth error (cm)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSETS / "breakdown.png")
    plt.close(fig)

    # 2. one calibration at 30 % liars, all four lines
    X, y, bad = calibration_run(0.30, np.random.default_rng(7))
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    ax.scatter(X[~bad, 0], y[~bad], s=10, color="#333", label="honest readings")
    ax.scatter(X[bad, 0], y[bad], s=14, marker="x", color="#b3261e", label="liars")
    xs = np.linspace(150, 1000, 2).reshape(-1, 1)
    for (name, make), c in zip(ESTIMATORS.items(), COLORS):
        ax.plot(xs, make().fit(X, y).predict(xs), color=c, lw=2, label=name)
    ax.set_xlabel("raw sensor counts")
    ax.set_ylabel("depth (m)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSETS / "thirty_percent.png")
    plt.close(fig)

    print("liars  " + "  ".join(f"{k:>20}" for k in err))
    for i, f in enumerate(FRACTIONS):
        print(f"{f*100:4.0f}%  " + "  ".join(f"{np.median(e[i]):17.1f} cm" for e in err.values()))
    for name, e in err.items():
        ok = [f for f, row in zip(FRACTIONS, e) if np.median(row) < 5.0]
        print(f"under 5 cm up to {max(ok)*100:.0f}% liars :: {name}" if ok else f"never under 5 cm :: {name}")


if __name__ == "__main__":
    main()
