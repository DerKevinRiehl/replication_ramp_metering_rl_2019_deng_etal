"""
Compose all space-time diagrams into a single grid image for presentation.
Layout: rows = scenarios (stationary, flat_peak, sharp_peak)
        cols = controllers (no_control, fixed_time, alinea, ppo)
Missing combinations shown as empty grey cells.
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import numpy as np

IMG_DIR = Path(__file__).parent.parent / "3_data_visualization" / "spacetime_diagrams"
OUT_DIR = IMG_DIR

SCENARIOS = ["stationary", "flat_peak", "sharp_peak"]
CONTROLLERS = ["no_control", "fixed_time", "alinea", "ppo"]

SCENARIO_LABELS = {
    "stationary": "Stationary",
    "flat_peak": "Flat Peak",
    "sharp_peak": "Sharp Peak",
}
CONTROLLER_LABELS = {
    "no_control": "No Control",
    "fixed_time": "Fixed Time",
    "alinea": "ALINEA",
    "ppo": "PPO",
}


def main():
    n_rows = len(SCENARIOS)
    n_cols = len(CONTROLLERS)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 5.5, n_rows * 2.8),
        gridspec_kw={"wspace": 0.05, "hspace": 0.15},
    )

    for r, scen in enumerate(SCENARIOS):
        for c, ctrl in enumerate(CONTROLLERS):
            ax = axes[r, c]
            fname = IMG_DIR / f"spacetime_{scen}_{ctrl}.png"
            if fname.exists():
                img = mpimg.imread(str(fname))
                ax.imshow(img)
            else:
                ax.set_facecolor("#e0e0e0")
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                        ha="center", va="center", fontsize=16, color="#999")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Column headers (top row only)
            if r == 0:
                ax.set_title(CONTROLLER_LABELS[ctrl], fontsize=14, fontweight="bold", pad=6)

            # Row labels (left column only)
            if c == 0:
                ax.set_ylabel(SCENARIO_LABELS[scen], fontsize=13, fontweight="bold",
                              labelpad=10, rotation=90)

    fig.suptitle("Space-Time Speed Diagrams — All Scenarios × Controllers",
                 fontsize=16, fontweight="bold", y=0.99)

    out_path = OUT_DIR / "spacetime_grid_all.png"
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
