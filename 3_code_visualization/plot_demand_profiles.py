"""
Plot demand profiles for the three scenarios.
Replicates Figure 5 from Deng et al. (2019).
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "1_code_produce"))

from config import SCENARIOS, SIMULATION_TIME, VIS_DATA_DIR
from utils import generate_demand_profile


def plot_demand_profiles():
    """
    Generate demand profile plots similar to Figure 5 in the paper.
    
    Creates a 2x3 subplot showing mainline and ramp demand for all scenarios.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    time_minutes = np.arange(0, SIMULATION_TIME // 60)
    
    scenario_titles = {
        'stationary': '(a/d) Stationary Volume',
        'flat_peak': '(b/e) Flat Peak',
        'sharp_peak': '(c/f) Sharp Peak'
    }
    
    for col, scenario in enumerate(SCENARIOS):
        # Mainline demand (top row)
        mainline_profile = generate_demand_profile(scenario, 'mainline')
        ax_main = axes[0, col]
        ax_main.plot(time_minutes, mainline_profile, 'b-', linewidth=2)
        ax_main.set_title(f"Mainline - {scenario_titles[scenario]}")
        ax_main.set_xlabel('Time (min)')
        ax_main.set_ylabel('Flow (veh/h)')
        ax_main.set_xlim([0, 120])
        ax_main.set_ylim([0, 8000])
        ax_main.grid(True, alpha=0.3)
        ax_main.axhline(y=6000, color='r', linestyle='--', alpha=0.5, label='Capacity')
        
        # Ramp demand (bottom row)
        ramp_profile = generate_demand_profile(scenario, 'ramp')
        ax_ramp = axes[1, col]
        ax_ramp.plot(time_minutes, ramp_profile, 'g-', linewidth=2)
        ax_ramp.set_title(f"Ramp - {scenario_titles[scenario]}")
        ax_ramp.set_xlabel('Time (min)')
        ax_ramp.set_ylabel('Flow (veh/h)')
        ax_ramp.set_xlim([0, 120])
        ax_ramp.set_ylim([0, 2000])
        ax_ramp.grid(True, alpha=0.3)
    
    plt.suptitle('Figure 5: Demand Profiles of Mainline and Ramp', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_path = VIS_DATA_DIR / 'demand_profiles.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved demand profiles to {output_path}")
    
    plt.show()


if __name__ == "__main__":
    plot_demand_profiles()
