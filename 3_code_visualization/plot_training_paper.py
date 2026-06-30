"""
Paper-ready PPO training figure (vector PDF), styled to match Deng et al. (2019), Fig. 7.

Reads the frozen per-episode training log (the exact run used in the paper) and
produces a 1x3 horizontal panel (Return | Average Speed | Average Queue Length),
matching the proportions, colours and labels of the original figure.

Output is a vector PDF (no rasterisation), suitable for direct inclusion in the
LaTeX article.

Usage:
    cd 3_code_visualization
    python plot_training_paper.py
    python plot_training_paper.py --csv path/to/log.csv --out path/to/fig.pdf
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CURVES_DIR = THIS_DIR.parent / "3_data_visualization" / "training_curves"

# Shared palette (kept consistent across all paper figures).
# Matches the original Deng et al. Fig. 7: return=red, speed=green, queue=blue.
PALETTE = {
    "return": {"raw": "#e74c3c", "line": "#c0392b"},
    "speed":  {"raw": "#2ecc71", "line": "#1e8449"},
    "queue":  {"raw": "#5dade2", "line": "#1a5276"},
}


def load_log(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    episodes: List[int] = []
    returns: List[float] = []
    speeds: List[float] = []
    queues: List[float] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(float(row["episode"])))
            returns.append(float(row["return"]))
            speeds.append(float(row["avg_speed_kmh"]))
            queues.append(float(row["avg_queue_m"]))
    return (
        np.asarray(episodes),
        np.asarray(returns, dtype=float),
        np.asarray(speeds, dtype=float),
        np.asarray(queues, dtype=float),
    )


def ema(values: np.ndarray, alpha: float = 0.9) -> np.ndarray:
    """Exponential moving average, NaN-safe."""
    out = np.empty_like(values)
    finite = values[np.isfinite(values)]
    last = finite[0] if finite.size else 0.0
    for i, v in enumerate(values):
        if np.isfinite(v):
            last = alpha * last + (1 - alpha) * v
        out[i] = last
    return out


def _panel(ax, x, y, key, ylabel):
    raw_c = PALETTE[key]["raw"]
    line_c = PALETTE[key]["line"]
    valid = np.isfinite(y)
    ax.plot(x[valid], y[valid], color=raw_c, alpha=0.30, linewidth=0.8)
    ax.plot(x, ema(y), color=line_c, linewidth=1.8)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.35, linewidth=0.6)
    ax.margins(x=0.01)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def make_figure(csv_path: Path, out_path: Path) -> None:
    episodes, returns, speeds, queues = load_log(csv_path)

    plt.rcParams.update({
        "font.family": "serif",
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.1))

    _panel(axes[0], episodes, returns, "return", "Return")
    _panel(axes[1], episodes, speeds, "speed", "Average Speed / km per hour")
    _panel(axes[2], episodes, queues, "queue", "Average Queue Length / m")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")          # vector PDF
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")  # preview
    plt.close(fig)

    print(f"Saved vector figure: {out_path}")
    print(f"  episodes: {len(episodes)}")
    print(f"  return : {returns[0]:.0f} -> {returns[-1]:.0f} (max {returns.max():.0f})")
    print(f"  speed  : {speeds[0]:.1f} -> {speeds[-1]:.1f} km/h")
    print(f"  queue  : {queues[0]:.1f} -> {queues[-1]:.1f} m")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-ready vector training figure")
    parser.add_argument("--scenario", "-s", default="flat_peak")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else DEFAULT_CURVES_DIR / f"{args.scenario}_training_log.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Training log CSV not found: {csv_path}")

    out_path = Path(args.out) if args.out else DEFAULT_CURVES_DIR / f"{args.scenario}_training_process.pdf"
    make_figure(csv_path, out_path)


if __name__ == "__main__":
    main()
