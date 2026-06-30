"""
Generate space-time speed diagrams of mainstream speed.
Replicates Figure 4 from Deng et al. (2019):
  x-axis = Time (minutes)
  y-axis = Detector index (p1 .. p5, spatial position upstream→downstream)
  color  = Mean speed (km/h)

Runs one episode per (scenario, controller) and saves a heatmap PNG.

Usage:
    python plot_spacetime.py --scenario stationary --controllers no_control fixed_time alinea
    python plot_spacetime.py --scenario all --controllers all
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    SCENARIOS, CONTROLLERS, CONTROL_STEPS, SIM_STEP,
    CONTROL_PERIOD_SECONDS, SIMULATION_TIME,
    DETECTOR_CONFIG, RANDOM_SEED, RESULTS_DIR,
    get_detector_ids, get_scenario_path
)
from sumo_env import SumoRampMeteringEnv
from controllers import (
    NoControlController,
    FixedTimeController,
    AlineaController,
    PPOController,
)
from utils import set_random_seed
import traci


# 30 visualization profiles (upstream → downstream, every ~33m)
N_VIS_PROFILES = 30
ALL_PROFILES = [f'vis_{i+1:02d}' for i in range(N_VIS_PROFILES)]


def _get_speed_limit_from_xml(scenario: str) -> float:
    """
    Read speed limits from scenario XML files and return max speed in km/h.

    Uses both:
      - Network.net.xml lane speeds
      - Configuration.rou.xml vehicle type maxSpeed
    """
    scenario_path = get_scenario_path(scenario)
    net_path = scenario_path / "Network.net.xml"
    rou_path = scenario_path / "Configuration.rou.xml"

    max_speed_ms = []

    if net_path.exists():
        try:
            tree = ET.parse(net_path)
            for lane in tree.findall(".//lane"):
                value = lane.get("speed")
                if value is not None:
                    max_speed_ms.append(float(value))
        except Exception:
            pass

    if rou_path.exists():
        try:
            tree = ET.parse(rou_path)
            for vtype in tree.findall(".//vType"):
                value = vtype.get("maxSpeed")
                if value is not None:
                    max_speed_ms.append(float(value))
        except Exception:
            pass

    if not max_speed_ms:
        return 80.0

    # Convert m/s to km/h and round up to nearest 5 for cleaner colorbar
    vmax_kmh = max(max_speed_ms) * 3.6
    vmax_kmh = 5.0 * np.ceil(vmax_kmh / 5.0)
    return float(vmax_kmh)


def get_controller(name, scenario):
    """Get controller instance."""
    if name == "no_control":
        return NoControlController()
    elif name == "fixed_time":
        return FixedTimeController(scenario=scenario)
    elif name == "alinea":
        return AlineaController(scenario=scenario)
    elif name == "ppo":
        from config import LOGS_DIR, get_model_path
        best_model_path = LOGS_DIR / "checkpoints" / "flat_peak" / "best_model.zip"
        if best_model_path.exists():
            model_path = best_model_path
        else:
            model_path = get_model_path("flat_peak")
            if not model_path.exists():
                model_path = get_model_path(scenario)
        if not model_path.exists():
            print(f"  PPO model not found, skipping.")
            return None
        return PPOController(model_path=str(model_path), scenario=scenario)
    return None


def collect_speed_timeseries(scenario, controller, seed=RANDOM_SEED):
    """
    Run one episode and collect per-step, per-profile mean speed (km/h).
    
    Returns:
        speed_matrix: np.ndarray of shape (n_profiles, n_steps) in km/h
        time_axis: np.ndarray of step times in seconds
    """
    env = SumoRampMeteringEnv(scenario=scenario, use_gui=False, add_noise=False)
    obs, _ = env.reset(seed=seed)
    controller.reset()

    label = env._conn_label

    # Build detector ID lists for all 30 visualization profiles
    # These are named det_vis_XX_lY in Additional.add.xml
    profile_detectors = {}
    for profile in ALL_PROFILES:
        # Get all detector IDs for this profile by querying TraCI
        # Format: det_vis_01_l0, det_vis_01_l1, det_vis_01_l2, ...
        traci.switch(label)
        all_loop_ids = traci.inductionloop.getIDList()
        prefix = f'det_{profile}_l'
        profile_detectors[profile] = [d for d in all_loop_ids if d.startswith(prefix)]

    speed_records = {p: [] for p in ALL_PROFILES}
    time_records = []

    done = False
    step_idx = 0
    while not done:
        action = controller.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        controller.step()

        # Current simulation time
        try:
            traci.switch(label)
            sim_time = traci.simulation.getTime()
        except:
            sim_time = step_idx * CONTROL_PERIOD_SECONDS

        time_records.append(sim_time)

        # Read mean speed from each profile (average across lanes)
        for profile in ALL_PROFILES:
            lane_speeds = []
            for det_id in profile_detectors[profile]:
                try:
                    traci.switch(label)
                    speed_ms = traci.inductionloop.getLastStepMeanSpeed(det_id)
                    if speed_ms >= 0:  # -1 means no vehicle
                        lane_speeds.append(speed_ms * 3.6)  # m/s → km/h
                except:
                    pass
            if lane_speeds:
                speed_records[profile].append(np.mean(lane_speeds))
            else:
                speed_records[profile].append(np.nan)

        step_idx += 1

    env.close()

    n_steps = len(time_records)
    speed_matrix = np.zeros((len(ALL_PROFILES), n_steps))
    for i, profile in enumerate(ALL_PROFILES):
        speed_matrix[i, :] = speed_records[profile]

    # Interpolate NaN values (detectors with no vehicles in a step)
    import pandas as pd
    df = pd.DataFrame(speed_matrix)
    # 1) Interpolate along time (axis=1) for each profile
    df = df.interpolate(axis=1, limit_direction='both')
    # 2) Interpolate along space (axis=0) for each timestep
    df = df.interpolate(axis=0, limit_direction='both')
    # 3) Fill any remaining NaN at edges with nearest value
    df = df.ffill(axis=1).bfill(axis=1)
    df = df.ffill(axis=0).bfill(axis=0)
    speed_matrix = df.values

    return speed_matrix, np.array(time_records)


def plot_spacetime(speed_matrix, time_axis, scenario, controller_name, save_dir):
    """
    Plot a space-time heatmap of mainstream speed.
    
    Args:
        speed_matrix: (n_profiles, n_steps) array of speeds in km/h
        time_axis: (n_steps,) array of times in seconds
        scenario: scenario name
        controller_name: controller name
        save_dir: directory to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    time_sec = time_axis  # already in seconds

    # Create colormap (paper uses blue-green-yellow-red style)
    cmap = plt.cm.RdYlGn  # Green=freeflow, Red=congested

    # Speed range: 0 to scenario max speed from XML (network + routes)
    vmin, vmax = 0.0, _get_speed_limit_from_xml(scenario)

    # Use imshow with extent for proper axis labels
    extent = [time_sec[0], time_sec[-1], -0.5, len(ALL_PROFILES) - 0.5]
    im = ax.imshow(
        speed_matrix,
        aspect='auto',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        origin='lower',
        interpolation='bicubic',
    )

    ax.set_xlabel('Time (s)', fontsize=13)
    ax.set_ylabel('Detector Index', fontsize=13)
    # Show tick marks at every 5th profile
    tick_positions = list(range(0, len(ALL_PROFILES), 5))
    tick_labels = [str(i+1) for i in tick_positions]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=11)
    ax.set_title(
        f'Space-Time Speed Diagram — {controller_name} ({scenario})',
        fontsize=14, fontweight='bold'
    )

    cbar = fig.colorbar(im, ax=ax, label='Mean Speed (km/h)', pad=0.02)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"spacetime_{scenario}_{controller_name}.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()
    return path


def main():
    parser = argparse.ArgumentParser(description="Generate space-time speed diagrams.")
    parser.add_argument('--scenario', '-s', nargs='+', default=['stationary'],
                        help="Scenarios (or 'all')")
    parser.add_argument('--controllers', '-c', nargs='+',
                        default=['no_control', 'fixed_time', 'alinea'],
                        help="Controllers (or 'all')")
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    scenarios = SCENARIOS if 'all' in args.scenario else args.scenario
    controllers = CONTROLLERS if 'all' in args.controllers else args.controllers

    set_random_seed(args.seed)

    save_dir = str(Path(__file__).parent.parent / '3_data_visualization' / 'spacetime_diagrams')

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"  Scenario: {scenario}")
        print(f"{'='*60}")
        for ctrl_name in controllers:
            print(f"\n  Controller: {ctrl_name}")
            ctrl = get_controller(ctrl_name, scenario)
            if ctrl is None:
                continue
            speed_matrix, time_axis = collect_speed_timeseries(
                scenario, ctrl, seed=args.seed
            )
            plot_spacetime(speed_matrix, time_axis, scenario, ctrl_name, save_dir)


if __name__ == '__main__':
    main()
