"""
Plot PPO Training Results from TensorBoard logs.

Generates learning curves for:
- Episode reward over time
- Episode length over time  
- Policy loss, value loss, entropy
- Learning rate and KL divergence

Usage:
    python plot_training_results.py --scenario flat_peak
    python plot_training_results.py --scenario flat_peak --logdir path/to/logs
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "1_code_produce"))
from config import LOGS_DIR, VIS_DATA_DIR

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("Error: tensorboard is required for plotting.")
    print("Install with: pip install tensorboard")
    sys.exit(1)


def load_tensorboard_logs(log_dir: Path) -> Dict[str, List[Tuple[int, float]]]:
    """
    Load data from TensorBoard event files.
    
    Args:
        log_dir: Directory containing TensorBoard logs
        
    Returns:
        Dictionary mapping metric names to (step, value) tuples
    """
    print(f"Loading TensorBoard logs from: {log_dir}")
    
    # Find all event files
    event_files = list(log_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {log_dir}")
    
    print(f"Found {len(event_files)} event file(s)")
    
    # Load all events
    ea = event_accumulator.EventAccumulator(str(log_dir))
    ea.Reload()
    
    # Extract scalar data
    data = {}
    scalar_tags = ea.Tags().get('scalars', [])
    
    print(f"Available metrics: {scalar_tags}")
    
    for tag in scalar_tags:
        events = ea.Scalars(tag)
        data[tag] = [(e.step, e.value) for e in events]
    
    return data


def smooth_curve(values: np.ndarray, weight: float = 0.9) -> np.ndarray:
    """
    Exponential moving average smoothing.
    
    Args:
        values: Array of values to smooth
        weight: Smoothing weight (0-1, higher = smoother)
        
    Returns:
        Smoothed values
    """
    smoothed = []
    last = values[0]
    for v in values:
        smoothed_val = last * weight + (1 - weight) * v
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)


def plot_training_curves(data: Dict[str, List[Tuple[int, float]]], 
                         scenario: str,
                         output_dir: Path):
    """
    Create 3 key training visualization plots.
    
    Args:
        data: Dictionary of metric data from TensorBoard
        scenario: Training scenario name
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = plt.cm.Set2.colors
    
    # ========== Plot 1: Episode Return ==========
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    if 'rollout/ep_rew_mean' in data:
        steps, rewards = zip(*data['rollout/ep_rew_mean'])
        steps = np.array(steps)
        rewards = np.array(rewards)
        
        # Plot raw and smoothed
        ax.plot(steps, rewards, alpha=0.3, color=colors[0], linewidth=0.8, label='Raw')
        ax.plot(steps, smooth_curve(rewards, 0.9), 
                color=colors[0], linewidth=2.5, label='Smoothed (EMA)')
        
        # Formatting
        ax.set_xlabel('Timesteps', fontsize=13, fontweight='bold')
        ax.set_ylabel('Episode Return', fontsize=13, fontweight='bold')
        ax.set_title(f'Training Return - {scenario.replace("_", " ").title()}', 
                    fontsize=15, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Final value annotation
        final_reward = rewards[-1]
        ax.axhline(y=final_reward, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
        ax.text(steps[-1]*0.98, final_reward*1.02, f'Final: {final_reward:.0f}',
                horizontalalignment='right', fontsize=10, bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = output_dir / f'{scenario}_return.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # ========== Plot 2: Average Speed (from eval callback) ==========
    # Note: TensorBoard doesn't log avg_speed by default, only from eval
    # We'll plot what's available or note it's missing
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    speed_key = None
    for key in data.keys():
        if 'speed' in key.lower() or 'eval' in key.lower():
            speed_key = key
            break
    
    if speed_key:
        steps, speeds = zip(*data[speed_key])
        steps = np.array(steps)
        speeds = np.array(speeds)
        
        ax.plot(steps, speeds, color=colors[1], linewidth=2.5, marker='o', markersize=4)
        ax.set_xlabel('Timesteps', fontsize=13, fontweight='bold')
        ax.set_ylabel('Average Speed (km/h)', fontsize=13, fontweight='bold')
        ax.set_title(f'Average Speed - {scenario.replace("_", " ").title()}', 
                    fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Speed data not available in TensorBoard logs\n(Only logged during evaluation)', 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_title(f'Average Speed - {scenario.replace("_", " ").title()}', 
                    fontsize=15, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / f'{scenario}_speed.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # ========== Plot 3: Average Queue (from eval callback) ==========
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    queue_key = None
    for key in data.keys():
        if 'queue' in key.lower():
            queue_key = key
            break
    
    if queue_key:
        steps, queues = zip(*data[queue_key])
        steps = np.array(steps)
        queues = np.array(queues)
        
        ax.plot(steps, queues, color=colors[2], linewidth=2.5, marker='s', markersize=4)
        ax.set_xlabel('Timesteps', fontsize=13, fontweight='bold')
        ax.set_ylabel('Average Queue Length (m)', fontsize=13, fontweight='bold')
        ax.set_title(f'Average Queue - {scenario.replace("_", " ").title()}', 
                    fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Queue data not available in TensorBoard logs\n(Only logged during evaluation)', 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_title(f'Average Queue - {scenario.replace("_", " ").title()}', 
                    fontsize=15, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / f'{scenario}_queue.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # ========== Summary Statistics ==========
    print("\n" + "="*60)
    print(f"TRAINING SUMMARY - {scenario.upper()}")
    print("="*60)
    
    if 'rollout/ep_rew_mean' in data:
        _, rewards = zip(*data['rollout/ep_rew_mean'])
        rewards = np.array(rewards)
        print(f"Initial Reward:  {rewards[0]:>10.2f}")
        print(f"Final Reward:    {rewards[-1]:>10.2f}")
        print(f"Max Reward:      {rewards.max():>10.2f}")
        print(f"Mean Reward:     {rewards.mean():>10.2f}")
        print(f"Std Reward:      {rewards.std():>10.2f}")
    
    if 'rollout/ep_len_mean' in data:
        _, lengths = zip(*data['rollout/ep_len_mean'])
        lengths = np.array(lengths)
        print(f"\nMean Ep Length:  {lengths.mean():>10.2f}")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Plot PPO training results')
    parser.add_argument('--scenario', type=str, default='flat_peak',
                       help='Training scenario (default: flat_peak)')
    parser.add_argument('--logdir', type=Path, default=None,
                       help='Path to TensorBoard logs directory (optional)')
    parser.add_argument('--output', type=Path, default=None,
                       help='Output directory for plots (optional)')
    
    args = parser.parse_args()
    
    # Determine log directory
    if args.logdir:
        log_dir = args.logdir
    else:
        # Find most recent log directory for scenario
        tensorboard_dir = LOGS_DIR / "tensorboard"
        if not tensorboard_dir.exists():
            print(f"Error: No tensorboard logs found at {tensorboard_dir}")
            sys.exit(1)
        
        # Find all matching logs
        matching_logs = sorted(tensorboard_dir.glob(f"{args.scenario}_*"))
        if not matching_logs:
            print(f"Error: No logs found for scenario '{args.scenario}'")
            print(f"Available logs in {tensorboard_dir}:")
            for log in tensorboard_dir.iterdir():
                if log.is_dir():
                    print(f"  - {log.name}")
            sys.exit(1)
        
        # Use most recent
        log_dir = matching_logs[-1]
        print(f"Using most recent log: {log_dir.name}")
        
        # Check if PPO_1 subdirectory exists (SB3 creates this)
        ppo_subdir = log_dir / "PPO_1"
        if ppo_subdir.exists():
            log_dir = ppo_subdir
            print(f"  Found PPO subdirectory: PPO_1")
    
    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = VIS_DATA_DIR / "training_curves"
    
    # Load and plot
    try:
        data = load_tensorboard_logs(log_dir)
        plot_training_curves(data, args.scenario, output_dir)
        print(f"\n✓ All plots saved to: {output_dir}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
