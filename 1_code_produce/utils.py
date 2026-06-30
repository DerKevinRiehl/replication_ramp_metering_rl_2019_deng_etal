"""
Utility functions for the ramp metering replication study.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt

from config import (
    SCENARIOS, RESULTS_DIR, VIS_DATA_DIR,
    MAINLINE_FLOW, RAMP_FLOW, SIMULATION_TIME
)


def generate_demand_profile(
    scenario: str,
    profile_type: str = 'mainline',
    add_noise: bool = False,
    noise_std: float = 0.05
) -> np.ndarray:
    """
    Generate demand profile for a scenario.

    The profiles are piecewise-constant intervals defined in config.py
    (extracted from the SUMO XML route files).

    Args:
        scenario: Scenario name ('stationary', 'flat_peak', 'sharp_peak')
        profile_type: 'mainline' or 'ramp'
        add_noise: Whether to add random noise
        noise_std: Standard deviation of noise (relative to flow)

    Returns:
        Array of flow values (veh/h) for each minute of simulation
    """
    num_minutes = SIMULATION_TIME // 60
    intervals = (MAINLINE_FLOW if profile_type == 'mainline' else RAMP_FLOW)[scenario]

    profile = np.zeros(num_minutes)
    for begin_s, end_s, per_hour in intervals:
        min_start = int(begin_s // 60)
        min_end = int(end_s // 60)
        # Clamp to simulation window
        min_start = max(0, min(min_start, num_minutes))
        min_end = max(0, min(min_end, num_minutes))
        profile[min_start:min_end] = per_hour

    # Add noise if requested
    if add_noise:
        noise = np.random.normal(0, noise_std * profile, size=profile.shape)
        profile = np.maximum(profile + noise, 0)

    return profile


def calculate_return(speeds: List[float], queues: List[float], eta: float = 0.1) -> float:
    """
    Calculate total return for an episode.
    
    Args:
        speeds: List of average speeds (km/h) per control period
        queues: List of queue lengths (m) per control period
        eta: Queue penalty coefficient
    
    Returns:
        Total return (sum of rewards)
    """
    rewards = [s - eta * q for s, q in zip(speeds, queues)]
    return sum(rewards)


def save_results(
    results: Dict[str, Dict[str, float]],
    scenario: str,
    output_path: Optional[Path] = None
) -> Path:
    """
    Save evaluation results to CSV.
    
    Args:
        results: Dictionary {controller_name: {metric: value}}
        scenario: Scenario name
        output_path: Optional output path
    
    Returns:
        Path to saved file
    """
    if output_path is None:
        output_path = RESULTS_DIR / f"{scenario}_results.csv"
    
    df = pd.DataFrame(results).T
    df.index.name = 'controller'
    df.to_csv(output_path)
    
    print(f"Results saved to {output_path}")
    return output_path


def load_results(scenario: str) -> pd.DataFrame:
    """
    Load evaluation results from CSV.
    
    Args:
        scenario: Scenario name
    
    Returns:
        DataFrame with results
    """
    path = RESULTS_DIR / f"{scenario}_results.csv"
    return pd.read_csv(path, index_col='controller')


def format_results_table(results: Dict[str, Dict[str, Dict[str, float]]]) -> pd.DataFrame:
    """
    Format results as Table I from the paper.
    
    Args:
        results: Nested dict {scenario: {controller: {metric: value}}}
    
    Returns:
        Formatted DataFrame
    """
    rows = []
    for scenario in SCENARIOS:
        for controller in results.get(scenario, {}):
            metrics = results[scenario][controller]
            rows.append({
                'Demand Profile': scenario.replace('_', ' ').title(),
                'Metering Strategy': controller.replace('_', ' ').title(),
                'Speed (km/h)': f"{metrics.get('avg_speed_kmh', 0):.2f}",
                'Queue size (m)': f"{metrics.get('avg_queue_m', 0):.2f}",
                'Return': f"{metrics.get('total_return', 0):.2f}",
            })
    
    return pd.DataFrame(rows)


def print_progress(
    episode: int,
    total_episodes: int,
    reward: float,
    avg_speed: float,
    avg_queue: float
):
    """Print training progress."""
    progress = episode / total_episodes * 100
    print(f"\rEpisode {episode}/{total_episodes} ({progress:.1f}%) | "
          f"Reward: {reward:.2f} | Speed: {avg_speed:.1f} km/h | "
          f"Queue: {avg_queue:.1f} m", end="")


def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


class EpisodeLogger:
    """Logger for tracking episode statistics."""
    
    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir
        self.episodes = []
        
    def log_episode(
        self,
        episode: int,
        controller: str,
        scenario: str,
        stats: Dict[str, float]
    ):
        """Log an episode's statistics."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'episode': episode,
            'controller': controller,
            'scenario': scenario,
            **stats
        }
        self.episodes.append(entry)
        
    def to_dataframe(self) -> pd.DataFrame:
        """Convert logged episodes to DataFrame."""
        return pd.DataFrame(self.episodes)
    
    def save(self, filename: str = "episode_log.csv"):
        """Save log to CSV."""
        if self.log_dir:
            path = self.log_dir / filename
        else:
            path = RESULTS_DIR / filename
        self.to_dataframe().to_csv(path, index=False)
        return path


def smooth_curve(values: List[float], window: int = 10) -> np.ndarray:
    """
    Smooth a curve using moving average.
    
    Args:
        values: Values to smooth
        window: Window size for moving average
    
    Returns:
        Smoothed values
    """
    if len(values) < window:
        return np.array(values)
    
    weights = np.ones(window) / window
    return np.convolve(values, weights, mode='valid')


def create_comparison_table(
    our_results: Dict[str, Dict[str, Dict[str, float]]],
    paper_results: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None
) -> pd.DataFrame:
    """
    Create comparison table between our results and paper results.
    
    Args:
        our_results: Our replication results
        paper_results: Original paper results (Table I)
    
    Returns:
        Comparison DataFrame
    """
    if paper_results is None:
        # Table I from paper
        paper_results = {
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
    
    rows = []
    for scenario in SCENARIOS:
        for controller in ['no_control', 'fixed_time', 'alinea', 'ppo']:
            paper = paper_results.get(scenario, {}).get(controller, {})
            ours = our_results.get(scenario, {}).get(controller, {})
            
            rows.append({
                'Scenario': scenario,
                'Controller': controller,
                'Paper Speed': paper.get('avg_speed_kmh', '-'),
                'Our Speed': f"{ours.get('avg_speed_kmh', 0):.2f}" if ours else '-',
                'Paper Queue': paper.get('avg_queue_m', '-'),
                'Our Queue': f"{ours.get('avg_queue_m', 0):.2f}" if ours else '-',
                'Paper Return': paper.get('total_return', '-'),
                'Our Return': f"{ours.get('total_return', 0):.2f}" if ours else '-',
            })
    
    return pd.DataFrame(rows)
