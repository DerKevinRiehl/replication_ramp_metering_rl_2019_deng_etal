"""
Create comparison table between our results and paper results.
Replicates Table I from Deng et al. (2019).
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "1_code_produce"))

from config import SCENARIOS, CONTROLLERS, RESULTS_DIR, VIS_DATA_DIR
from utils import load_results, create_comparison_table


# Paper results from Table I
PAPER_RESULTS = {
    'stationary': {
        'no_control': {'avg_speed_kmh': 38.62, 'avg_queue_m': 4.57, 'total_return': 4579},
        'fixed_time': {'avg_speed_kmh': 47.84, 'avg_queue_m': 39.49, 'total_return': 5266},
        'alinea': {'avg_speed_kmh': 57.31, 'avg_queue_m': 118.81, 'total_return': 5450},
        'ppo': {'avg_speed_kmh': 55.24, 'avg_queue_m': 92.26, 'total_return': 5521},
    },
    'flat_peak': {
        'no_control': {'avg_speed_kmh': 43.08, 'avg_queue_m': 4.45, 'total_return': 5116},
        'fixed_time': {'avg_speed_kmh': 51.49, 'avg_queue_m': 37.28, 'total_return': 5731},
        'alinea': {'avg_speed_kmh': 49.89, 'avg_queue_m': 38.74, 'total_return': 5522},
        'ppo': {'avg_speed_kmh': 55.71, 'avg_queue_m': 28.32, 'total_return': 6345},
    },
    'sharp_peak': {
        'no_control': {'avg_speed_kmh': 54.66, 'avg_queue_m': 1.47, 'total_return': 6541},
        'fixed_time': {'avg_speed_kmh': 59.76, 'avg_queue_m': 38.37, 'total_return': 6711},
        'alinea': {'avg_speed_kmh': 57.54, 'avg_queue_m': 11.09, 'total_return': 6771},
        'ppo': {'avg_speed_kmh': 60.97, 'avg_queue_m': 29.34, 'total_return': 6964},
    },
}


def plot_table_comparison():
    """
    Create a visual comparison table showing paper vs replicated results.
    """
    # Try to load our results
    our_results = {}
    for scenario in SCENARIOS:
        try:
            df = load_results(scenario)
            our_results[scenario] = df.to_dict('index')
        except FileNotFoundError:
            print(f"Results not found for {scenario}, using paper values")
            our_results[scenario] = {}
    
    # Create figure with table
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    for idx, scenario in enumerate(SCENARIOS):
        ax = axes[idx]
        ax.axis('off')
        
        # Prepare table data
        table_data = []
        columns = ['Metering Strategy', 
                   'Paper Speed', 'Our Speed',
                   'Paper Queue', 'Our Queue',
                   'Paper Return', 'Our Return']
        
        for controller in ['no_control', 'fixed_time', 'alinea', 'ppo']:
            paper = PAPER_RESULTS[scenario][controller]
            ours = our_results[scenario].get(controller, {})
            
            row = [
                controller.replace('_', ' ').title(),
                f"{paper['avg_speed_kmh']:.2f}",
                f"{ours.get('avg_speed_kmh', '-'):.2f}" if ours else '-',
                f"{paper['avg_queue_m']:.2f}",
                f"{ours.get('avg_queue_m', '-'):.2f}" if ours else '-',
                f"{paper['total_return']:.0f}",
                f"{ours.get('total_return', '-'):.0f}" if ours else '-',
            ]
            table_data.append(row)
        
        table = ax.table(
            cellText=table_data,
            colLabels=columns,
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style header
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(color='white', fontweight='bold')
        
        # Highlight best returns
        best_paper_idx = np.argmax([PAPER_RESULTS[scenario][c]['total_return'] 
                                    for c in ['no_control', 'fixed_time', 'alinea', 'ppo']])
        table[(best_paper_idx + 1, 5)].set_facecolor('#C6EFCE')
        
        ax.set_title(f"{scenario.replace('_', ' ').title()} Demand Profile", 
                    fontsize=12, fontweight='bold', pad=20)
    
    plt.suptitle('Table I: Comparison of Metering Strategies\n(Paper vs Replication)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = VIS_DATA_DIR / 'table_I_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison table to {output_path}")
    
    plt.show()


def print_comparison_table():
    """Print text-based comparison table."""
    print("\n" + "=" * 80)
    print("TABLE I: COMPARISON OF METERING STRATEGIES")
    print("=" * 80)
    
    for scenario in SCENARIOS:
        print(f"\n{scenario.replace('_', ' ').upper()} DEMAND PROFILE")
        print("-" * 80)
        print(f"{'Strategy':<15} {'Speed (km/h)':<25} {'Queue (m)':<25} {'Return':<15}")
        print(f"{'':15} {'Paper':<12} {'Ours':<12} {'Paper':<12} {'Ours':<12} {'Paper':<7} {'Ours':<7}")
        print("-" * 80)
        
        for controller in ['no_control', 'fixed_time', 'alinea', 'ppo']:
            paper = PAPER_RESULTS[scenario][controller]
            
            # Try to load our results
            try:
                df = load_results(scenario)
                ours = df.loc[controller].to_dict() if controller in df.index else {}
            except:
                ours = {}
            
            print(f"{controller.replace('_', ' ').title():<15} "
                  f"{paper['avg_speed_kmh']:<12.2f} {ours.get('avg_speed_kmh', '-')!s:<12} "
                  f"{paper['avg_queue_m']:<12.2f} {ours.get('avg_queue_m', '-')!s:<12} "
                  f"{paper['total_return']:<7.0f} {ours.get('total_return', '-')!s:<7}")


if __name__ == "__main__":
    print_comparison_table()
    plot_table_comparison()
