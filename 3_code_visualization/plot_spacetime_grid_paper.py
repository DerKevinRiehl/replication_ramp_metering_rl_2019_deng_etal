"""
Paper-ready space-time speed grid (vector PDF), styled to match Deng et al. (2019), Fig. 8.

Builds a single 3x4 figure:
    rows    = demand scenarios (Stationary, Flat Peak, Sharp Peak)
    columns = controllers       (No Control, Fixed-Time, ALINEA, PPO)
with ONE shared colorbar (no per-panel colorbars / titles), matching the
original figure's layout, proportions and colour scale.

CACHING
-------
Re-rendering the grid does not require SUMO. The per-cell speed matrices are
cached as .npz files in `3_data_visualization/spacetime_cache/`. The first run
(on a machine with SUMO) computes and caches them; subsequent runs (anywhere)
re-plot instantly from the cache, so design tweaks are SUMO-free.

Usage:
    # first run, on a machine with SUMO (computes + caches + plots):
    python plot_spacetime_grid_paper.py
    # later, anywhere (re-plot from cache only):
    python plot_spacetime_grid_paper.py --from-cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PRODUCE_DIR = THIS_DIR.parent / "1_code_produce"
VIS_DIR = THIS_DIR.parent / "3_data_visualization"
CACHE_DIR = VIS_DIR / "spacetime_cache"

SCENARIOS = ["stationary", "flat_peak", "sharp_peak"]
CONTROLLERS = ["no_control", "fixed_time", "alinea", "ppo"]

SCENARIO_LABELS = {"stationary": "Stationary", "flat_peak": "Flat Peak", "sharp_peak": "Sharp Peak"}
CONTROLLER_LABELS = {"no_control": "No Control", "fixed_time": "Fixed-Time", "alinea": "ALINEA", "ppo": "PPO"}

PANEL_LETTERS = list("abcdefghijkl")


def cache_path(scenario: str, controller: str) -> Path:
    return CACHE_DIR / f"{scenario}_{controller}.npz"


def compute_cell(scenario: str, controller: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Run one SUMO episode and return (speed_matrix, time_axis). Requires SUMO."""
    sys.path.insert(0, str(PRODUCE_DIR))
    from plot_spacetime import collect_speed_timeseries, get_controller  # lazy: needs SUMO

    ctrl = get_controller(controller, scenario)
    if ctrl is None:
        raise RuntimeError(f"Controller '{controller}' unavailable (missing model?).")
    speed_matrix, time_axis = collect_speed_timeseries(scenario, ctrl, seed=seed)
    return speed_matrix, time_axis


def get_cell(scenario: str, controller: str, seed: int, from_cache: bool) -> tuple[np.ndarray, np.ndarray]:
    cp = cache_path(scenario, controller)
    if cp.exists():
        d = np.load(cp)
        return d["speed_matrix"], d["time_axis"]
    if from_cache:
        raise FileNotFoundError(f"Cache missing and --from-cache set: {cp}")
    speed_matrix, time_axis = compute_cell(scenario, controller, seed)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cp, speed_matrix=speed_matrix, time_axis=time_axis)
    print(f"  cached: {cp.name}")
    return speed_matrix, time_axis


def make_grid(out_path: Path, seed: int, from_cache: bool, vmax: float | None) -> None:
    # ---- gather data ----
    cells = {}
    for scen in SCENARIOS:
        for ctrl in CONTROLLERS:
            print(f"[{scen:>11} | {ctrl:>10}] loading ...")
            cells[(scen, ctrl)] = get_cell(scen, ctrl, seed, from_cache)

    # Shared colour scale across ALL panels (required for one shared colorbar).
    if vmax is None:
        vmax = max(np.nanmax(m) for m, _ in cells.values())
        vmax = float(10.0 * np.ceil(vmax / 10.0))
    vmin = 0.0

    plt.rcParams.update({"font.family": "serif", "xtick.labelsize": 8, "ytick.labelsize": 8})

    n_rows, n_cols = len(SCENARIOS), len(CONTROLLERS)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(13, 6.6),
        gridspec_kw={"wspace": 0.06, "hspace": 0.12},
    )

    cmap = plt.cm.RdYlGn
    im = None
    k = 0
    for r, scen in enumerate(SCENARIOS):
        for c, ctrl in enumerate(CONTROLLERS):
            ax = axes[r, c]
            speed_matrix, time_axis = cells[(scen, ctrl)]
            n_prof = speed_matrix.shape[0]
            extent = [time_axis[0], time_axis[-1], -0.5, n_prof - 0.5]
            im = ax.imshow(
                speed_matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                extent=extent, origin="lower", interpolation="bicubic",
            )
            # panel letter, bottom-right (like the original)
            ax.text(0.97, 0.06, f"({PANEL_LETTERS[k]})", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=10, color="white", fontweight="bold")
            k += 1

            if r == 0:
                ax.set_title(CONTROLLER_LABELS[ctrl], fontsize=12, fontweight="bold", pad=6)
            if c == 0:
                ax.set_ylabel(f"{SCENARIO_LABELS[scen]}\nDetector Index", fontsize=10)
                ax.set_yticks(range(0, n_prof, 5))
                ax.set_yticklabels([str(i) for i in range(0, n_prof, 5)])
            else:
                ax.set_yticks([])
            if r == n_rows - 1:
                ax.set_xlabel("Time / s", fontsize=10)
            else:
                ax.set_xticks([])

    # ---- single shared colorbar ----
    fig.subplots_adjust(right=0.90)
    cbar_ax = fig.add_axes([0.92, 0.12, 0.015, 0.76])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r"Speed / km$\cdot$h$^{-1}$", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")                      # vector PDF
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")  # preview
    plt.close(fig)
    print(f"\nSaved vector grid: {out_path}  (vmax={vmax:.0f})")


def main() -> None:
    p = argparse.ArgumentParser(description="Paper-ready vector space-time grid")
    p.add_argument("--from-cache", action="store_true", help="Only plot from cached .npz (no SUMO)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vmax", type=float, default=None, help="Fix colour-scale max (km/h)")
    p.add_argument("--out", default=str(VIS_DIR / "spacetime_diagrams" / "spacetime_grid_paper.pdf"))
    args = p.parse_args()
    make_grid(Path(args.out), seed=args.seed, from_cache=args.from_cache, vmax=args.vmax)


if __name__ == "__main__":
    main()
