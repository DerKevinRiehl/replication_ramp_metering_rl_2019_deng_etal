"""
Paper-ready multi-ramp per-period figure (vector PDF), styled to match
Deng et al. (2019), Fig. 9.

Builds a single 1x3 figure:
    Reward | Speed (km/h) | Queue (m)   vs Control Period
comparing ALINEA (blue) and MAPPO (orange), with
    ramp 1 = solid line, ramp 2 = dashed line,
mean over evaluation episodes and a +/- std shaded band, and a shared bottom
legend (Method + Ramp Index) like the original.

CACHING
-------
Re-rendering does not require SUMO. Per-period, per-ramp traces are cached as a
single .npz in `3_data_visualization/`. The first run (machine with SUMO)
computes + caches; later runs (anywhere) re-plot from cache.

Usage:
    # first run, on a machine with SUMO:
    python plot_multiramp_paper.py --episodes 10
    # later, anywhere:
    python plot_multiramp_paper.py --from-cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PRODUCE_DIR = THIS_DIR.parent / "1_code_produce"
VIS_DIR = THIS_DIR.parent / "3_data_visualization"
CACHE = VIS_DIR / "multiramp_period_cache.npz"

METHODS = ["alinea", "mappo"]
METHOD_LABELS = {"alinea": "ALINEA", "mappo": "Proposed Method (MAPPO)"}
METHOD_COLORS = {"alinea": "#3b6fb0", "mappo": "#d9722e"}  # blue / orange, as original
RAMP_STYLE = {1: "-", 2: "--"}

# trace keys produced by evaluate_all_marl.run_episode
METRICS = [
    ("Reward", "reward"),
    ("Speed (km/h)", "speed"),
    ("Queue Size (m)", "queue"),
]


def collect(episodes: int, seed: int) -> dict:
    """Run SUMO episodes for ALINEA and MAPPO; return stacked per-ramp traces."""
    sys.path.insert(0, str(PRODUCE_DIR))
    from evaluate_multi_ramp import run_episode  # needs SUMO

    data = {}
    for method in METHODS:
        per_key = {f"{m}_r{r}": [] for _, m in METRICS for r in (1, 2)}
        for ep in range(episodes):
            _stats, traces = run_episode(controller=method, seed=seed + ep, use_gui=False)
            for _, m in METRICS:
                per_key[f"{m}_r1"].append(traces[f"{m}_r1"])
                per_key[f"{m}_r2"].append(traces[f"{m}_r2"])
            print(f"  {method:>6} ep {ep}: {len(traces['reward_r1'])} periods")
        for k in per_key:
            data[f"{method}__{k}"] = np.asarray(per_key[k], dtype=float)  # (n_ep, n_periods)
    return data


def make_figure(out_path: Path, data: dict) -> None:
    plt.rcParams.update({"font.family": "serif", "xtick.labelsize": 9, "ytick.labelsize": 9})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    for ax, (ylabel, m) in zip(axes, METRICS):
        for method in METHODS:
            color = METHOD_COLORS[method]
            for ramp in (1, 2):
                arr = data[f"{method}__{m}_r{ramp}"]      # (n_ep, n_periods)
                mean = np.nanmean(arr, axis=0)
                std = np.nanstd(arr, axis=0)
                x = np.arange(len(mean))
                ax.plot(x, mean, color=color, linestyle=RAMP_STYLE[ramp], linewidth=1.8)
                if arr.shape[0] > 1:
                    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlabel("Control Period", fontsize=11)
        ax.grid(True, alpha=0.3, linewidth=0.6)
        ax.margins(x=0.01)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    # shared bottom legend: Method (colour) + Ramp Index (linestyle)
    method_handles = [Patch(facecolor=METHOD_COLORS[mn], label=METHOD_LABELS[mn]) for mn in METHODS]
    ramp_handles = [
        Line2D([0], [0], color="0.3", linestyle=RAMP_STYLE[1], label="Ramp 1"),
        Line2D([0], [0], color="0.3", linestyle=RAMP_STYLE[2], label="Ramp 2"),
    ]
    fig.legend(handles=method_handles + ramp_handles, loc="lower center",
               ncol=4, frameon=True, fontsize=10, edgecolor="#cccccc",
               bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")                      # vector PDF
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")  # preview
    plt.close(fig)
    print(f"\nSaved vector figure: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Paper-ready vector multi-ramp period figure")
    p.add_argument("--from-cache", action="store_true", help="Only plot from cached .npz (no SUMO)")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(VIS_DIR / "spacetime_diagrams" / "multiramp_period_paper.pdf"))
    args = p.parse_args()

    if CACHE.exists():
        print(f"Loading cache: {CACHE}")
        data = {k: v for k, v in np.load(CACHE).items()}
    elif args.from_cache:
        raise FileNotFoundError(f"Cache missing and --from-cache set: {CACHE}")
    else:
        data = collect(args.episodes, args.seed)
        np.savez_compressed(CACHE, **data)
        print(f"Cached traces: {CACHE}")

    make_figure(Path(args.out), data)


if __name__ == "__main__":
    main()
