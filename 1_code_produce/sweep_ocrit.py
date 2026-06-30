"""
Sweep ALINEA O_crit and KR to find the values that best match the
paper targets for each demand scenario (minimises combined normalised
error on speed and queue).

Paper targets (Table I):
  stationary : speed=57.31, queue=118.81
  flat_peak  : speed=49.89, queue= 38.74
  sharp_peak : speed=57.54, queue= 11.09

Usage:
    python sweep_ocrit.py                               # full 2-D grid, all scenarios
    python sweep_ocrit.py --scenario flat_peak
    python sweep_ocrit.py --ocrit 0.10 0.12 0.14 0.16  # custom O_crit range
    python sweep_ocrit.py --kr 0.20 0.35 0.50           # custom KR range
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from sumo_env import SumoRampMeteringEnv
from controllers.alinea import AlineaController

# Paper reference values per scenario
PAPER_TARGETS = {
    'stationary': {'speed': 57.31, 'queue': 118.81, 'ret': 5450},
    'flat_peak':  {'speed': 49.89, 'queue':  38.74, 'ret': 5522},
    'sharp_peak': {'speed': 57.54, 'queue':  11.09, 'ret': 6771},
}

DEFAULT_OCRIT = [round(x, 2) for x in np.arange(0.04, 0.16, 0.01)]
DEFAULT_KR    = [round(x, 2) for x in np.arange(0.10, 0.65, 0.05)]  # 0.10..0.60 step 0.05


def combined_error(speed, queue, target):
    """Normalised combined error: |dSpeed|/v* + |dQueue|/q* (equal weight)."""
    dv = abs(speed - target['speed']) / target['speed']
    dq = abs(queue - target['queue']) / target['queue']
    return dv + dq


def run_episode(scenario, o_crit, k_r, seed=42):
    env = SumoRampMeteringEnv(scenario=scenario, use_gui=False)
    ctrl = AlineaController(o_crit=o_crit, k_r=k_r)
    obs, _ = env.reset(seed=seed)
    ctrl.reset()
    done = False
    while not done:
        action = ctrl.get_action(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        ctrl.step()
    stats = env.get_episode_statistics()
    env.close()
    return stats['avg_speed_kmh'], stats['avg_queue_m'], stats['total_return']


def sweep_scenario(scenario, o_crit_values, kr_values):
    target = PAPER_TARGETS[scenario]
    n_runs = len(o_crit_values) * len(kr_values)
    print(f"\n{'='*75}")
    print(f"Scenario: {scenario}  |  Target -> speed={target['speed']:.2f}  queue={target['queue']:.2f}  ({n_runs} runs)")
    print(f"{'='*75}")

    header = f"{'O_crit':>7}  {'KR':>5}  {'Speed':>8}  {'Queue':>8}  {'Return':>10}  {'|dV|%':>7}  {'|dQ|%':>7}  {'err':>7}"
    print(header)
    print("-" * len(header))

    best = {'err': float('inf'), 'o_crit': None, 'k_r': None, 'speed': None, 'queue': None, 'ret': None}

    for k_r in kr_values:
        for oc in o_crit_values:
            speed, queue, ret = run_episode(scenario, oc, k_r)
            err = combined_error(speed, queue, target)
            dv_pct = abs(speed - target['speed']) / target['speed'] * 100
            dq_pct = abs(queue - target['queue']) / target['queue'] * 100
            marker = " <--" if err < best['err'] else ""
            if err < best['err']:
                best = {'err': err, 'o_crit': oc, 'k_r': k_r, 'speed': speed, 'queue': queue, 'ret': ret}
            print(f"{oc:7.2f}  {k_r:5.2f}  {speed:8.2f}  {queue:8.2f}  {ret:10.2f}  {dv_pct:7.1f}  {dq_pct:7.1f}  {err:7.4f}{marker}")

    print(f"\n>>> Best for {scenario}: O_crit={best['o_crit']:.2f}, KR={best['k_r']:.2f}  "
          f"(speed={best['speed']:.2f} vs {target['speed']:.2f}, "
          f"queue={best['queue']:.2f} vs {target['queue']:.2f})")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', '-s', nargs='+',
                        default=['stationary', 'flat_peak', 'sharp_peak'],
                        help="Scenarios to sweep (default: all)")
    parser.add_argument('--ocrit', nargs='+', type=float,
                        default=DEFAULT_OCRIT,
                        help="O_crit values to try")
    parser.add_argument('--kr', nargs='+', type=float,
                        default=DEFAULT_KR,
                        help="KR values to try (default: 0.10..0.60 step 0.05)")
    args = parser.parse_args()

    results = {}
    for sc in args.scenario:
        results[sc] = sweep_scenario(sc, args.ocrit, args.kr)

    print("\n" + "="*65)
    print("SUMMARY — recommended config values:")
    print("="*65)
    for sc, best in results.items():
        print(f"  {sc:<12}: ALINEA_O_CRIT = {best['o_crit']:.2f},  ALINEA_KR = {best['k_r']:.2f}")


if __name__ == "__main__":
    main()
