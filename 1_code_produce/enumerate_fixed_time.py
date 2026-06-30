"""
Enumerate fixed-time metering rates to match the paper's grid search.

Usage examples:
  python enumerate_fixed_time.py --scenario flat_peak
  python enumerate_fixed_time.py --scenario all --rate-min 200 --rate-max 1200 --rate-step 50
  python enumerate_fixed_time.py --scenario sharp_peak --rates 300,400,500,600
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# Paper reference values for fixed-time control (Table I)
PAPER_FIXED_TIME = {
    'stationary': {'speed': 47.84, 'queue': 39.49, 'return': 5266},
    'flat_peak':  {'speed': 51.49, 'queue': 37.28, 'return': 5731},
    'sharp_peak': {'speed': 59.76, 'queue': 38.37, 'return': 6711},
}

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    SCENARIOS, EVAL_EPISODES, RANDOM_SEED, RESULTS_DIR
)
from sumo_env import SumoRampMeteringEnv
from controllers import FixedTimeController
from utils import set_random_seed


def evaluate_fixed_time(scenario: str, rate: float, num_episodes: int, seed: int):
    env = SumoRampMeteringEnv(scenario=scenario, use_gui=False, add_noise=False)
    controller = FixedTimeController(scenario=scenario, metering_rate=rate)

    returns = []
    speeds = []
    queues = []

    for episode in range(num_episodes):
        observation, _ = env.reset(seed=seed + episode)
        controller.reset()
        done = False

        while not done:
            action = controller.get_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            controller.step()

        stats = env.get_episode_statistics()
        returns.append(stats['total_return'])
        speeds.append(stats['avg_speed_kmh'])
        queues.append(stats['avg_queue_m'])

    env.close()

    return {
        "scenario": scenario,
        "metering_rate": rate,
        "total_return": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "avg_speed_kmh": float(np.mean(speeds)),
        "speed_std": float(np.std(speeds)),
        "avg_queue_m": float(np.mean(queues)),
        "queue_std": float(np.std(queues)),
        "num_episodes": num_episodes,
    }


def parse_rates(args):
    if args.rates:
        return [float(x.strip()) for x in args.rates.split(",") if x.strip()]
    return list(np.arange(args.rate_min, args.rate_max + 1e-9, args.rate_step))


def main():
    parser = argparse.ArgumentParser(description="Enumerate fixed-time metering rates.")
    parser.add_argument("--scenario", default="flat_peak",
                        help="Scenario to evaluate (stationary, flat_peak, sharp_peak, all)")
    parser.add_argument("--rates", default=None,
                        help="Comma-separated list of rates (veh/h). Overrides min/max/step.")
    parser.add_argument("--rate-min", type=float, default=200.0)
    parser.add_argument("--rate-max", type=float, default=1200.0)
    parser.add_argument("--rate-step", type=float, default=50.0)
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--metric", default="paper_distance",
                        choices=["avg_speed_kmh", "total_return", "avg_queue_m", "paper_distance"],
                        help="Metric to optimize. paper_distance minimizes combined error vs paper.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)

    args = parser.parse_args()

    scenarios = SCENARIOS if args.scenario == "all" else [args.scenario]
    rates = parse_rates(args)

    set_random_seed(args.seed)

    for scenario in scenarios:
        print(f"\nScenario: {scenario}")
        print(f"Rates: {rates[0]} .. {rates[-1]} (n={len(rates)})")

        results = []
        for rate in tqdm(rates, desc=f"fixed_time/{scenario}"):
            results.append(evaluate_fixed_time(scenario, rate, args.episodes, args.seed))

        df = pd.DataFrame(results)

        # Compute paper distance for each row
        if scenario in PAPER_FIXED_TIME:
            paper = PAPER_FIXED_TIME[scenario]
            df['paper_distance'] = df.apply(
                lambda r: abs(r['avg_speed_kmh'] - paper['speed']) / paper['speed']
                        + abs(r['avg_queue_m'] - paper['queue']) / paper['queue'],
                axis=1
            )

        # Select best rate
        if args.metric in ("avg_queue_m", "paper_distance"):
            best_row = df.loc[df[args.metric].idxmin()]
        else:
            best_row = df.loc[df[args.metric].idxmax()]

        cols = ["metering_rate", "avg_speed_kmh", "total_return", "avg_queue_m"]
        if 'paper_distance' in df.columns:
            cols.append('paper_distance')
        print(f"\nBest fixed-time rate (metric={args.metric}):")
        print(best_row[cols])

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / f"fixed_time_enumeration_{scenario}.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved results to: {out_path}")


if __name__ == "__main__":
    main()
