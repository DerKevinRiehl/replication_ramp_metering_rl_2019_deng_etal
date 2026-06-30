"""
Parameter sweep to find the configuration closest to the paper's no-control results.

Tests combinations of:
  - departSpeed: "avg" (current) vs "0" (Flow default)
  - departLane: per-lane "0/1/2" vs "0" only (all rightmost, Flow default) 
  - tau: 1.5 (current) vs 1.6
  - sigma: 0.5 (current) vs 0.7

Target: speed=38.62 km/h, queue=4.57 m (paper stationary no-control)

Runs 1 episode per config on stationary/no_control scenario.
"""

import sys
import os
import shutil
import tempfile
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_SOURCE_DIR, SIMULATION_TIME, CONTROL_PERIOD_SECONDS,
    SIM_STEP, NUM_CONTROL_PERIODS, CONTROL_STEPS,
    MR_MIN, MR_MAX, RESULTS_DIR
)


# ── Parameter grid ──────────────────────────────────────────────
PARAM_GRID = {
    'departSpeed': ['avg', '0'],           # current vs Flow default
    'departLane_mode': ['per_lane', 'all_rightmost'],  # 3 flows vs 1 flow on lane 0
    'tau': [1.5, 1.6],                     # current vs slightly higher
    'sigma': [0.5, 0.7],                   # current vs more imperfection
}

# Paper targets
PAPER_SPEED = 38.62
PAPER_QUEUE = 4.57
PAPER_RETURN = 4579


def generate_rou_xml(depart_speed, depart_lane_mode, tau, sigma):
    """Generate a temporary .rou.xml with the given parameters."""
    
    # Stationary: mainline 6000 veh/h (3 lanes), ramp 1000 veh/h (1 lane)
    main_prob = round(6000 / 3 / 3600, 4)   # 0.5556
    ramp_prob = round(1000 / 1 / 3600, 4)   # 0.2778
    
    vtype = (
        f'    <vType id="DEFAULT_VEHTYPE" accel="2.6" decel="4.5" sigma="{sigma}"\n'
        f'        length="5.0" minGap="2.5" maxSpeed="22.22" tau="{tau}"\n'
        f'        speedDev="0.1" lcCooperative="0.5" lcAssertive="1.0"/>'
    )
    
    flows = []
    
    if depart_lane_mode == 'per_lane':
        # One flow per lane (current approach)
        for lane in range(3):
            flows.append(
                f'    <flow type="DEFAULT_VEHTYPE" id="flow_main_L{lane}" '
                f'begin="0.00" end="7200.00" probability="{main_prob}" '
                f'departLane="{lane}" departSpeed="{depart_speed}" '
                f'from="E_upstream" to="E_main_after"/>'
            )
    else:
        # All vehicles on rightmost lane (Flow default)
        # Need 3x probability on the single lane to match total demand
        all_right_prob = round(6000 / 1 / 3600, 4)  # 1.6667
        # probability > 1 means ~1.67 vehicles per second attempt
        # SUMO caps at 1.0 for probability — use vehsPerHour instead
        # Actually, probability > 1 is valid in SUMO (it spawns multiple per step)
        # But safer to use vehsPerHour for this case
        flows.append(
            f'    <flow type="DEFAULT_VEHTYPE" id="flow_main_L0" '
            f'begin="0.00" end="7200.00" vehsPerHour="6000" '
            f'departLane="0" departSpeed="{depart_speed}" '
            f'from="E_upstream" to="E_main_after"/>'
        )
    
    # Ramp (always 1 lane)
    flows.append(
        f'    <flow type="DEFAULT_VEHTYPE" id="flow_ramp_L0" '
        f'begin="0.00" end="7200.00" probability="{ramp_prob}" '
        f'departLane="0" departSpeed="max" '
        f'color="255,22,0" from="E_ramp" to="E_main_after"/>'
    )
    
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n'
        f'{vtype}\n'
        + '\n'.join(flows) + '\n'
        '</routes>\n'
    )
    return xml


def run_single_episode(rou_xml_content, scenario_dir):
    """Run 1 episode of no-control on stationary and return metrics."""
    
    # Write temporary rou file
    rou_path = os.path.join(scenario_dir, 'Configuration.rou.xml')
    original_content = open(rou_path, 'r').read()
    
    try:
        with open(rou_path, 'w') as f:
            f.write(rou_xml_content)
        
        # Import here to avoid traci conflicts
        from sumo_env import SumoRampMeteringEnv
        from controllers import NoControlController
        
        env = SumoRampMeteringEnv(
            scenario='stationary',
            use_gui=False,
            add_noise=False,
        )
        controller = NoControlController()
        
        obs, _ = env.reset(seed=42)
        controller.reset()
        done = False
        
        while not done:
            action = controller.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            controller.step()
        
        stats = env.get_episode_statistics()
        env.close()
        
        return {
            'speed': stats['avg_speed_kmh'],
            'queue': stats['avg_queue_m'],
            'return': stats['total_return'],
        }
    finally:
        # Restore original file
        with open(rou_path, 'w') as f:
            f.write(original_content)


def compute_distance(speed, queue):
    """Normalized distance from paper results."""
    d_speed = abs(speed - PAPER_SPEED) / PAPER_SPEED
    d_queue = abs(queue - PAPER_QUEUE) / PAPER_QUEUE
    return d_speed + d_queue


def main():
    scenario_dir = str(DATA_SOURCE_DIR / 'stationary')
    
    # Save original rou file
    rou_path = os.path.join(scenario_dir, 'Configuration.rou.xml')
    with open(rou_path, 'r') as f:
        original_rou = f.read()
    
    # Generate all combinations
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combinations = list(itertools.product(*values))
    
    print(f"{'='*80}")
    print(f"Parameter Sweep: {len(combinations)} configurations")
    print(f"Target: speed={PAPER_SPEED} km/h, queue={PAPER_QUEUE} m")
    print(f"{'='*80}")
    
    results = []
    
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        depart_speed = params['departSpeed']
        depart_lane_mode = params['departLane_mode']
        tau = params['tau']
        sigma = params['sigma']
        
        label = f"dS={depart_speed:>3s} dL={depart_lane_mode:>13s} τ={tau} σ={sigma}"
        print(f"\n[{i+1}/{len(combinations)}] {label}")
        
        # Generate and run
        rou_xml = generate_rou_xml(depart_speed, depart_lane_mode, tau, sigma)
        metrics = run_single_episode(rou_xml, scenario_dir)
        
        d = compute_distance(metrics['speed'], metrics['queue'])
        
        print(f"  → speed={metrics['speed']:.2f} km/h  queue={metrics['queue']:.2f} m  "
              f"return={metrics['return']:.0f}  d_paper={d:.3f}")
        
        results.append({
            'departSpeed': depart_speed,
            'departLane': depart_lane_mode,
            'tau': tau,
            'sigma': sigma,
            'speed_kmh': round(metrics['speed'], 2),
            'queue_m': round(metrics['queue'], 2),
            'total_return': round(metrics['return'], 0),
            'd_paper': round(d, 4),
        })
    
    # Restore original
    with open(rou_path, 'w') as f:
        f.write(original_rou)
    
    # Sort by distance
    df = pd.DataFrame(results).sort_values('d_paper')
    
    print(f"\n{'='*80}")
    print("RESULTS (sorted by distance from paper)")
    print(f"{'='*80}")
    print(f"Paper reference: speed={PAPER_SPEED} km/h, queue={PAPER_QUEUE} m, return={PAPER_RETURN}")
    print()
    print(df.to_string(index=False))
    
    # Save
    out_path = RESULTS_DIR / 'param_sweep_depart.csv'
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")
    
    # Highlight best
    best = df.iloc[0]
    print(f"\n{'='*80}")
    print(f"BEST CONFIG (d_paper={best['d_paper']:.4f}):")
    print(f"  departSpeed={best['departSpeed']}, departLane={best['departLane']}, "
          f"tau={best['tau']}, sigma={best['sigma']}")
    print(f"  speed={best['speed_kmh']} km/h (paper: {PAPER_SPEED})")
    print(f"  queue={best['queue_m']} m (paper: {PAPER_QUEUE})")
    print(f"  return={best['total_return']} (paper: {PAPER_RETURN})")


if __name__ == '__main__':
    main()
