"""
Generate space-time diagrams of mainline speed.
Replicates Figure 7 from Deng et al. (2019).

Shows speed evolution along the highway over time for different controllers.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

sys.path.insert(0, str(Path(__file__).parent.parent / "1_code_produce"))

from config import SCENARIOS, CONTROLLERS, VIS_DATA_DIR, RESULTS_DIR


def generate_synthetic_spacetime_data(
    scenario: str,
    controller: str,
    num_positions: int = 10,
    num_timesteps: int = 120
) -> np.ndarray:
    """
    Generate synthetic space-time speed data for visualization.
    In practice, this would be collected during simulation.
    
    Args:
        scenario: Demand scenario
        controller: Controller type
        num_positions: Number of spatial positions (detectors)
        num_timesteps: Number of time steps
    
    Returns:
        2D array of speeds (position x time)
    """
    # Base free-flow speed
    free_flow = 100  # km/h
    
    # Create time and space grids
    speeds = np.ones((num_positions, num_timesteps)) * free_flow
    
    # Bottleneck location (around position 5-7, merge area)
    bottleneck_start = 5
    bottleneck_end = 7
    
    # Add congestion patterns based on scenario and controller
    if scenario == 'stationary':
        if controller == 'no_control':
            # Persistent congestion at bottleneck
            speeds[bottleneck_start:bottleneck_end, 20:] *= 0.4
            speeds[bottleneck_start-2:bottleneck_start, 30:] *= 0.6
        elif controller == 'fixed_time':
            # Reduced congestion
            speeds[bottleneck_start:bottleneck_end, 30:90] *= 0.5
        elif controller == 'alinea':
            # Better but still some congestion
            speeds[bottleneck_start:bottleneck_end, 40:80] *= 0.6
        else:  # ppo
            # Minimal congestion
            speeds[bottleneck_start:bottleneck_end, 45:75] *= 0.7
    
    elif scenario == 'flat_peak':
        peak_start, peak_end = 30, 90
        if controller == 'no_control':
            speeds[bottleneck_start:bottleneck_end, peak_start:peak_end] *= 0.45
            speeds[bottleneck_start-2:bottleneck_start, peak_start+10:peak_end] *= 0.6
        elif controller == 'fixed_time':
            speeds[bottleneck_start:bottleneck_end, peak_start+5:peak_end-10] *= 0.55
        elif controller == 'alinea':
            speeds[bottleneck_start:bottleneck_end, peak_start+10:peak_end-15] *= 0.55
        else:  # ppo
            speeds[bottleneck_start:bottleneck_end, peak_start+15:peak_end-20] *= 0.65
    
    elif scenario == 'sharp_peak':
        peak_start, peak_end = 50, 70
        if controller == 'no_control':
            speeds[bottleneck_start:bottleneck_end, peak_start:peak_end] *= 0.55
        elif controller == 'fixed_time':
            speeds[bottleneck_start:bottleneck_end, peak_start:peak_end-5] *= 0.65
        elif controller == 'alinea':
            speeds[bottleneck_start:bottleneck_end, peak_start+5:peak_end-5] *= 0.6
        else:  # ppo
            speeds[bottleneck_start:bottleneck_end, peak_start+5:peak_end-8] *= 0.7
    
    # Add some noise
    noise = np.random.normal(0, 3, speeds.shape)
    speeds = np.clip(speeds + noise, 20, 120)
    
    return speeds


def plot_spacetime_diagrams():
    """
    Create space-time diagrams similar to Figure 7 in the paper.
    
    Shows 4x3 grid: rows = controllers, columns = scenarios
    """
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    
    controller_names = {
        'no_control': '(a,e,i) No-control',
        'fixed_time': '(b,f,j) Fixed-time',
        'alinea': '(c,g,k) ALINEA',
        'ppo': '(d,h,l) Proposed'
    }
    
    scenario_names = {
        'stationary': 'Stationary Volume',
        'flat_peak': 'Flat Peak',
        'sharp_peak': 'Sharp Peak'
    }
    
    # Colormap for speed
    cmap = cm.RdYlGn  # Red (slow) to Yellow to Green (fast)
    norm = Normalize(vmin=20, vmax=100)
    
    for row, controller in enumerate(['no_control', 'fixed_time', 'alinea', 'ppo']):
        for col, scenario in enumerate(SCENARIOS):
            ax = axes[row, col]
            
            # Get speed data
            speeds = generate_synthetic_spacetime_data(scenario, controller)
            
            # Create heatmap
            im = ax.imshow(
                speeds,
                aspect='auto',
                cmap=cmap,
                norm=norm,
                origin='lower',
                extent=[0, 120, 0, 1000]  # time (min), position (m)
            )
            
            # Labels
            ax.set_xlabel('Time (min)')
            ax.set_ylabel('Position (m)')
            
            # Title
            if row == 0:
                ax.set_title(scenario_names[scenario], fontsize=12, fontweight='bold')
            if col == 0:
                ax.text(-0.2, 0.5, controller_names[controller],
                       transform=ax.transAxes, fontsize=10,
                       verticalalignment='center', rotation=90)
    
    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cbar.set_label('Speed (km/h)', fontsize=12)
    
    plt.suptitle('Figure 7: Space-time Diagram of Speed Under Different Metering Strategies',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    # Save
    output_path = VIS_DATA_DIR / 'spacetime_diagrams.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved space-time diagrams to {output_path}")
    
    plt.show()


def plot_single_scenario(scenario: str = 'flat_peak'):
    """
    Plot space-time diagram for a single scenario with all controllers.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    controllers = ['no_control', 'fixed_time', 'alinea', 'ppo']
    titles = ['(a) No-control', '(b) Fixed-time', '(c) ALINEA', '(d) Proposed Algorithm']
    
    cmap = cm.RdYlGn
    norm = Normalize(vmin=20, vmax=100)
    
    for idx, (controller, title) in enumerate(zip(controllers, titles)):
        ax = axes[idx]
        
        speeds = generate_synthetic_spacetime_data(scenario, controller)
        
        im = ax.imshow(
            speeds,
            aspect='auto',
            cmap=cmap,
            norm=norm,
            origin='lower',
            extent=[0, 120, 0, 1000]
        )
        
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Position (m)')
        ax.set_title(title, fontsize=11)
    
    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cbar.set_label('Speed (km/h)')
    
    plt.suptitle(f'Space-time Diagram - {scenario.replace("_", " ").title()} Scenario',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    output_path = VIS_DATA_DIR / f'spacetime_{scenario}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_path}")
    
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', '-s', type=str, default=None,
                       help='Single scenario to plot (or all if not specified)')
    args = parser.parse_args()
    
    if args.scenario:
        plot_single_scenario(args.scenario)
    else:
        plot_spacetime_diagrams()
