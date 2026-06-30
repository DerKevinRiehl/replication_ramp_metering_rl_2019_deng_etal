"""
Evaluation Script for All Controllers.
Evaluates all four control strategies across all demand scenarios,
replicating the results in Table I of Deng et al. (2019).

Usage:
    python evaluate_all.py --scenarios all
    python evaluate_all.py --scenarios flat_peak --controllers ppo alinea
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    SCENARIOS, CONTROLLERS, EVAL_EPISODES, RANDOM_SEED,
    RESULTS_DIR, get_model_path
)
from sumo_env import SumoRampMeteringEnv
from controllers import (
    NoControlController,
    FixedTimeController,
    AlineaController,
)
from utils import (
    save_results, format_results_table, 
    create_comparison_table, set_random_seed, EpisodeLogger
)


def get_controller(name: str, scenario: str):
    """
    Get controller instance by name.
    
    Args:
        name: Controller name
        scenario: Scenario for controller configuration
    
    Returns:
        Controller instance
    """
    if name == "no_control":
        return NoControlController()
    elif name == "fixed_time":
        return FixedTimeController(scenario=scenario)
    elif name == "alinea":
        return AlineaController(scenario=scenario)
    elif name == "ppo":
        from controllers.ppo_controller import PPOController

        # Paper Section III.C.1: PPO is trained on flat_peak only,
        # then tested on all 3 scenarios to demonstrate generalization.
        
        # Try best_model.zip from checkpoints first (best performance during training)
        from pathlib import Path
        from config import LOGS_DIR
        best_model_path = LOGS_DIR / "checkpoints" / "flat_peak" / "best_model.zip"
        
        if best_model_path.exists():
            model_path = best_model_path
        else:
            # Fallback to default model path
            model_path = get_model_path("flat_peak")
            if not model_path.exists():
                # Second fallback: try scenario-specific model
                model_path = get_model_path(scenario)
        
        if not model_path.exists():
            print(f"Warning: PPO model not found at {model_path}")
            print("Train a model first with: python train_ppo.py --scenario flat_peak")
            return None
        
        print(f"  Loading PPO model from: {model_path.name}")
        return PPOController(model_path=str(model_path), scenario=scenario)
    else:
        raise ValueError(f"Unknown controller: {name}")


def evaluate_controller(
    controller,
    scenario: str,
    num_episodes: int = EVAL_EPISODES,
    seed: int = RANDOM_SEED,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate a controller on a scenario.
    
    Args:
        controller: Controller instance
        scenario: Traffic scenario
        num_episodes: Number of evaluation episodes
        seed: Random seed
        verbose: Whether to show progress
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Create environment (no noise during evaluation)
    env = SumoRampMeteringEnv(
        scenario=scenario,
        use_gui=False,
        add_noise=False,
    )
    
    all_returns = []
    all_speeds = []
    all_queues = []
    # Per-detector speed stats
    detector_profiles = ['p2', 'p3', 'p4', 'p5']
    all_detector_speeds = {k: [] for k in detector_profiles}
    
    episodes = range(num_episodes)
    if verbose:
        episodes = tqdm(episodes, desc=f"  {controller.name}")

    all_vehicle_flows = []
    for episode in episodes:
        # Reset environment and controller
        observation, _ = env.reset(seed=seed + episode)
        controller.reset()

        episode_return = 0.0
        episode_speeds = []
        episode_queues = []
        done = False

        while not done:
            # Get action from controller
            action = controller.get_action(observation)

            # Step environment
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_return += reward
            episode_speeds.append(info.get('avg_speed', 0))
            episode_queues.append(info.get('avg_queue', 0))

            controller.step()

        # Get final episode statistics
        stats = env.get_episode_statistics()
        all_returns.append(stats['total_return'])
        all_speeds.append(stats['avg_speed_kmh'])
        all_queues.append(stats['avg_queue_m'])
        # Per-detector speeds
        for k in detector_profiles:
            key = f'avg_speed_{k}_kmh'
            if key in stats:
                all_detector_speeds[k].append(stats[key])
        # Collect vehicle flows for this episode
        all_vehicle_flows.extend(stats.get('vehicle_flows', []))
    env.close()
    
    # Aggregate results
    results = {
        'total_return': np.mean(all_returns),
        'return_std': np.std(all_returns),
        'avg_speed_kmh': np.mean(all_speeds),
        'speed_std': np.std(all_speeds),
        'avg_queue_m': np.mean(all_queues),
        'queue_std': np.std(all_queues),
        'num_episodes': num_episodes,
    }
    # Add per-detector means and stds
    for k in detector_profiles:
        arr = np.array(all_detector_speeds[k])
        results[f'avg_speed_{k}_kmh'] = float(np.mean(arr)) if len(arr) else 0.0
        results[f'speed_{k}_std'] = float(np.std(arr)) if len(arr) else 0.0
    # Add vehicle flows for plotting
    results['vehicle_flows'] = all_vehicle_flows
    return results


def evaluate_all(
    scenarios: List[str] = None,
    controllers: List[str] = None,
    num_episodes: int = EVAL_EPISODES,
    seed: int = RANDOM_SEED,
    save: bool = True,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Evaluate all controllers on all scenarios.
    
    Args:
        scenarios: List of scenarios (default: all)
        controllers: List of controllers (default: all)
        num_episodes: Episodes per evaluation
        seed: Random seed
        save: Whether to save results
    
    Returns:
        Nested dict {scenario: {controller: {metric: value}}}
    """
    if scenarios is None:
        scenarios = SCENARIOS
    if controllers is None:
        controllers = CONTROLLERS
    
    print("=" * 60)
    print("Ramp Metering Controller Evaluation")
    print("Replicating Table I from Deng et al. (2019)")
    print("=" * 60)
    print(f"Scenarios: {scenarios}")
    print(f"Controllers: {controllers}")
    print(f"Episodes per evaluation: {num_episodes}")
    print("=" * 60)
    
    set_random_seed(seed)

    all_results = {}
    
    for scenario in scenarios:
        print(f"\n{'='*40}")
        print(f"Scenario: {scenario.replace('_', ' ').title()}")
        print(f"{'='*40}")
        
        scenario_results = {}
        
        for controller_name in controllers:
            print(f"\nEvaluating {controller_name}...")
            
            # Get controller
            controller = get_controller(controller_name, scenario)
            if controller is None:
                print(f"  Skipping {controller_name} (not available)")
                continue
            
            # Evaluate
            results = evaluate_controller(
                controller=controller,
                scenario=scenario,
                num_episodes=num_episodes,
                seed=seed,
                verbose=True,
            )
            
            scenario_results[controller_name] = results

            # Print summary
            print(f"  Results:")
            print(f"    Speed: {results['avg_speed_kmh']:.2f} ± {results['speed_std']:.2f} km/h")
            print(f"    Queue: {results['avg_queue_m']:.2f} ± {results['queue_std']:.2f} m")
            print(f"    Return: {results['total_return']:.2f} ± {results['return_std']:.2f}")
            # Print per-detector speeds
            for k in ['p2', 'p3', 'p4', 'p5']:
                print(f"    Speed {k}: {results[f'avg_speed_{k}_kmh']:.2f} ± {results[f'speed_{k}_std']:.2f} km/h")

            # Plot flow diagram after simulation
            vehicle_flows = results.get('vehicle_flows', [])
            if vehicle_flows:
                import os
                from config import DATA_SOURCE_DIR
                from measure_demand import analyze_and_plot_demand

                vis_dir = os.path.join(os.path.dirname(__file__), '..', '3_data_visualization')
                analyze_and_plot_demand(scenario, controller_name, vehicle_flows, DATA_SOURCE_DIR, vis_dir)
        
        all_results[scenario] = scenario_results
        
        # Save scenario results
        if save:
            save_results(scenario_results, scenario)
    
    # Print comparison table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (Table I Comparison)")
    print("=" * 60)
    
    comparison_df = create_comparison_table(all_results)
    print(comparison_df.to_string(index=False))
    
    # Save comparison table
    if save:
        comparison_path = RESULTS_DIR / "comparison_with_paper.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\nComparison saved to: {comparison_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ramp metering controllers (Table I replication)"
    )
    parser.add_argument(
        "--scenarios", "-s",
        nargs="+",
        default=["stationary"],
        help="Scenarios to evaluate (or 'all')"
    )
    parser.add_argument(
        "--controllers", "-c",
        nargs="+",
        default=["all"],
        help="Controllers to evaluate (or 'all')"
    )
    parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=EVAL_EPISODES,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results"
    )
    
    args = parser.parse_args()
    
    # Parse scenarios
    scenarios = SCENARIOS if "all" in args.scenarios else args.scenarios
    
    # Parse controllers
    controllers = CONTROLLERS if "all" in args.controllers else args.controllers
    
    # Run evaluation
    evaluate_all(
        scenarios=scenarios,
        controllers=controllers,
        num_episodes=args.episodes,
        seed=args.seed,
        save=not args.no_save,
    )


if __name__ == "__main__":
    main()
