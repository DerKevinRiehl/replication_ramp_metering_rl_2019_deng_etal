"""
Debug evaluation script - runs ONE episode per controller with verbose output.
Prints observations, actions, metering rates, and TLS states at each step.

Usage:
    python debug_eval.py --scenario flat_peak
    python debug_eval.py --scenario stationary --controllers alinea fixed_time
"""

import argparse
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    NUM_CONTROL_PERIODS, MR_MIN, MR_MAX,
    GREEN_PHASE, DETECTOR_PROFILES, DOWNSTREAM_PROFILE,
    TLS_ID, get_detector_ids, QUEUE_DETECTOR_ID
)
from sumo_env import SumoRampMeteringEnv
from controllers import (
    NoControlController, FixedTimeController,
    AlineaController, PPOController,
)
from evaluate_all import get_controller
import traci


def debug_one_episode(controller, scenario: str, max_steps: int = 20):
    """Run one episode with detailed debug output."""
    
    print(f"\n{'#'*70}")
    print(f"# Controller: {controller.name} | Scenario: {scenario}")
    print(f"{'#'*70}")
    
    env = SumoRampMeteringEnv(
        scenario=scenario,
        use_gui=False,
        add_noise=False,
    )
    
    obs, info = env.reset(seed=42)
    controller.reset()
    
    print(f"\n--- Initial State ---")
    print(f"  Observation (occupancies): {obs}")
    print(f"  Profiles: {DETECTOR_PROFILES}")
    print(f"  Downstream profile: {DOWNSTREAM_PROFILE}")
    
    # Check TLS initial state
    traci.switch(env._conn_label)
    try:
        tls_state = traci.trafficlight.getRedYellowGreenState(TLS_ID)
        tls_program = traci.trafficlight.getProgram(TLS_ID)
        tls_phase = traci.trafficlight.getPhase(TLS_ID)
        print(f"  TLS state: {tls_state}, program: {tls_program}, phase: {tls_phase}")
    except Exception as e:
        print(f"  TLS error: {e}")
    
    total_return = 0.0
    all_speeds = []
    all_queues = []
    all_actions = []
    all_mr = []
    
    steps_to_print = min(max_steps, NUM_CONTROL_PERIODS)
    
    for step in range(NUM_CONTROL_PERIODS):
        # Get action from controller
        action = controller.get_action(obs)
        
        # Convert to metering rate (same logic as env)
        action_val = np.clip(action[0], 0.0, 1.0)
        metering_rate = MR_MIN + action_val * (MR_MAX - MR_MIN)
        
        # Compute red duration (same logic as env)
        if metering_rate <= 0:
            red_float = 57  # MAX_RED
        else:
            cycle_time = 3600.0 / metering_rate
            red_float = cycle_time - GREEN_PHASE
            red_float = np.clip(red_float, 0, 57)
        red_duration = max(1, round(red_float)) if metering_rate < MR_MAX else 0
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        controller.step()
        
        total_return += reward
        
        # Read actual detector values
        traci.switch(env._conn_label)
        
        # Get individual detector occupancies for downstream
        downstream_occs = []
        for det_id in get_detector_ids(DOWNSTREAM_PROFILE):
            try:
                occ = traci.inductionloop.getLastIntervalOccupancy(det_id)
                downstream_occs.append(occ)
            except:
                downstream_occs.append(-1)
        
        # Get speed at downstream
        downstream_speeds = []
        for det_id in get_detector_ids(DOWNSTREAM_PROFILE):
            try:
                spd = traci.inductionloop.getLastIntervalMeanSpeed(det_id)
                downstream_speeds.append(spd * 3.6 if spd >= 0 else -1)
            except:
                downstream_speeds.append(-1)
        
        # Get queue
        try:
            queue = traci.lanearea.getJamLengthMeters(QUEUE_DETECTOR_ID)
        except:
            queue = -1
        
        # Get TLS state
        try:
            tls_state = traci.trafficlight.getRedYellowGreenState(TLS_ID)
            tls_phase = traci.trafficlight.getPhase(TLS_ID)
        except:
            tls_state = "?"
            tls_phase = -1
        
        # Get number of vehicles on ramp
        try:
            ramp_vehs = traci.edge.getLastStepVehicleNumber("E_ramp")
            ramp_halting = traci.edge.getLastStepHaltingNumber("E_ramp")
        except:
            ramp_vehs = -1
            ramp_halting = -1
        
        speed_kmh = np.mean([s for s in downstream_speeds if s >= 0]) if any(s >= 0 for s in downstream_speeds) else 0
        all_speeds.append(speed_kmh)
        all_queues.append(queue)
        all_actions.append(action_val)
        all_mr.append(metering_rate)
        
        if step < steps_to_print or step == NUM_CONTROL_PERIODS - 1:
            print(f"\n--- Step {step+1}/{NUM_CONTROL_PERIODS} ---")
            print(f"  Obs (occupancies):     {obs}")
            print(f"  Action (normalized):   {action_val:.4f}")
            print(f"  Metering rate:         {metering_rate:.1f} veh/h")
            print(f"  Red duration:          {red_duration}s (green={GREEN_PHASE}s, cycle={GREEN_PHASE+red_duration}s)")
            print(f"  TLS state: '{tls_state}', phase: {tls_phase}")
            print(f"  Downstream occ (%):    {downstream_occs}")
            print(f"  Downstream speed km/h: {[f'{s:.1f}' for s in downstream_speeds]}")
            print(f"  Queue on ramp:         {queue:.1f} m")
            print(f"  Ramp vehicles:         {ramp_vehs} (halting: {ramp_halting})")
            print(f"  Reward:                {reward:.2f}")
            print(f"  Cumulative return:     {total_return:.2f}")
        elif step == steps_to_print:
            print(f"\n  ... (skipping steps {steps_to_print+1}-{NUM_CONTROL_PERIODS-1}, printing last step) ...")
        
        if terminated or truncated:
            break
    
    stats = env.get_episode_statistics()
    env.close()
    
    print(f"\n{'='*50}")
    print(f"  EPISODE SUMMARY: {controller.name} / {scenario}")
    print(f"{'='*50}")
    print(f"  Avg Speed:      {stats['avg_speed_kmh']:.2f} km/h")
    print(f"  Avg Queue:      {stats['avg_queue_m']:.2f} m")
    print(f"  Total Return:   {stats['total_return']:.2f}")
    print(f"  Action range:   [{min(all_actions):.4f}, {max(all_actions):.4f}]")
    print(f"  MR range:       [{min(all_mr):.1f}, {max(all_mr):.1f}] veh/h")
    print(f"  Speed range:    [{min(all_speeds):.1f}, {max(all_speeds):.1f}] km/h")
    print(f"  Queue range:    [{min(all_queues):.1f}, {max(all_queues):.1f}] m")
    
    # For ALINEA, print extra info
    if hasattr(controller, 'current_rate'):
        print(f"  ALINEA final rate: {controller.current_rate:.1f} veh/h")
        print(f"  ALINEA o_crit:     {controller.o_crit}")
        print(f"  ALINEA K_R:        {controller.k_r}")
        print(f"  ALINEA downstream_idx: {controller.downstream_idx}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Debug evaluation")
    parser.add_argument("--scenario", "-s", default="flat_peak")
    parser.add_argument("--controllers", "-c", nargs="+", 
                       default=["no_control", "fixed_time", "alinea"])
    parser.add_argument("--steps", type=int, default=15,
                       help="Max steps to print in detail")
    args = parser.parse_args()
    
    print(f"MR_MIN = {MR_MIN:.1f} veh/h, MR_MAX = {MR_MAX:.1f} veh/h")
    print(f"GREEN_PHASE = {GREEN_PHASE}s")
    print(f"DETECTOR_PROFILES = {DETECTOR_PROFILES}")
    print(f"DOWNSTREAM_PROFILE = {DOWNSTREAM_PROFILE}")
    
    for ctrl_name in args.controllers:
        ctrl = get_controller(ctrl_name, args.scenario)
        if ctrl is not None:
            debug_one_episode(ctrl, args.scenario, max_steps=args.steps)


if __name__ == "__main__":
    main()
