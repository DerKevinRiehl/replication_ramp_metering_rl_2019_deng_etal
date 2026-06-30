"""
Regenerate the PPO training-process figure from a saved training_log.csv.

WHY THIS EXISTS
---------------
The training curve in the paper was produced by ONE specific run of
`train_ppo.py`, whose per-episode data was saved to
`3_data_visualization/training_curves/<scenario>_training_log.csv`.
The PNG was lost, but the CSV is the ground truth for that run.

Re-running `train_ppo.py` does NOT reproduce the paper figure: it trains a
brand-new stochastic agent (different seed schedule, library versions, SUMO
non-determinism), so the curve will differ. To get *the paper figure back*,
plot directly from the frozen CSV.

This script does not import SUMO or stable-baselines3, so it runs anywhere.

Usage:
    cd 3_code_visualization
    python regenerate_training_plot.py
    python regenerate_training_plot.py --scenario flat_peak
    python regenerate_training_plot.py --csv path/to/log.csv --out path/to/fig.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CURVES_DIR = THIS_DIR.parent / "3_data_visualization" / "training_curves"


def load_log(csv_path: Path):
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
        np.array(episodes),
        np.array(returns, dtype=float),
        np.array(speeds, dtype=float),
        np.array(queues, dtype=float),
    )


def ema(values: np.ndarray, alpha: float = 0.95) -> np.ndarray:
    """Exponential moving average, NaN-safe. Matches train_ppo.plot_training_process."""
    out = np.empty_like(values)
    finite = values[np.isfinite(values)]
    last = finite[0] if finite.size else 0.0
    for i, v in enumerate(values):
        if np.isfinite(v):
            last = alpha * last + (1 - alpha) * v
        out[i] = last
    return out


def regenerate(csv_path: Path, out_path: Path, scenario: str) -> None:
    episodes, returns, speeds, queues = load_log(csv_path)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(
        f'PPO Training Process — {scenario.replace("_", " ").title()}\n'
        f"(cf. Deng et al. 2019, Fig. 7)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    ax = axes[0]
    ax.plot(episodes, returns, color="#e74c3c", alpha=0.25, linewidth=0.8, label="Raw")
    ax.plot(episodes, ema(returns), color="#c0392b", linewidth=2.0, label="Smoothed (EMA)")
    ax.set_ylabel("Episode Return", fontsize=12, fontweight="bold")
    ax.set_title("(a) Episode Return", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    valid = np.isfinite(speeds)
    ax.plot(episodes[valid], speeds[valid], color="#27ae60", alpha=0.25, linewidth=0.8, label="Raw")
    ax.plot(episodes, ema(speeds), color="#1e8449", linewidth=2.0, label="Smoothed (EMA)")
    ax.set_ylabel("Avg Speed (km/h)", fontsize=12, fontweight="bold")
    ax.set_title("(b) Average Bottleneck Speed", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    valid = np.isfinite(queues)
    ax.plot(episodes[valid], queues[valid], color="#2980b9", alpha=0.25, linewidth=0.8, label="Raw")
    ax.plot(episodes, ema(queues), color="#1a5276", linewidth=2.0, label="Smoothed (EMA)")
    ax.set_ylabel("Avg Queue Length (m)", fontsize=12, fontweight="bold")
    ax.set_title("(c) Average Ramp Queue Length", fontsize=13)
    ax.set_xlabel("Episode", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Regenerated training figure from CSV: {out_path}")
    print(f"  episodes: {len(episodes)}")
    print(f"  return  : first={returns[0]:.1f}  last={returns[-1]:.1f}  max={returns.max():.1f}")
    print(f"  speed   : first={speeds[0]:.1f}  last={speeds[-1]:.1f} km/h")
    print(f"  queue   : first={queues[0]:.1f}  last={queues[-1]:.1f} m")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate training plot from saved CSV (no SUMO needed)")
    parser.add_argument("--scenario", "-s", default="flat_peak")
    parser.add_argument("--csv", default=None, help="Path to <scenario>_training_log.csv")
    parser.add_argument("--out", default=None, help="Output PNG path")
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else DEFAULT_CURVES_DIR / f"{args.scenario}_training_log.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Training log CSV not found: {csv_path}\n"
            "This file is the frozen record of the run used in the paper. "
            "If it is missing, the exact paper curve cannot be reproduced without re-training."
        )

    out_path = Path(args.out) if args.out else DEFAULT_CURVES_DIR / f"{args.scenario}_training_process.png"
    regenerate(csv_path, out_path, args.scenario)


if __name__ == "__main__":
    main()
