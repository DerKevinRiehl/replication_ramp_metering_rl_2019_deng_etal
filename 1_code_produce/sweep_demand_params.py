import os
import sys
import xml.etree.ElementTree as ET
import itertools
from copy import deepcopy
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import traci

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_SOURCE_DIR, RESULTS_DIR, SIM_STEP, RANDOM_SEED
from sumo_env import SumoRampMeteringEnv
from controllers import NoControlController
from measure_demand import parse_xml_demand
from utils import set_random_seed

# Define the grid of parameters to test
PARAM_GRID = {
    'departSpeed': ['avg', 'max'],
    'departPos': ['base', 'free', 'last'],
    'departLane': ['0', 'best'],
    'tau': ['1.0', '1.5'],
    'minGap': ['1.0', '2.5']
}

def create_sweep_xml(scenario: str, params: dict) -> str:
    """Read the original Configuration.rou.xml, apply parameters, and save as a new file."""
    scenario_dir = os.path.join(DATA_SOURCE_DIR, scenario)
    orig_file = os.path.join(scenario_dir, 'Configuration.rou.xml')
    sweep_file = os.path.join(scenario_dir, 'Configuration_sweep.rou.xml')
    
    tree = ET.parse(orig_file)
    root = tree.getroot()
    
    # 1. Update vType parameters (tau, minGap)
    for vtype in root.findall('vType'):
        if 'tau' in params:
            vtype.set('tau', params['tau'])
        if 'minGap' in params:
            vtype.set('minGap', params['minGap'])
            
    # 2. Update flow parameters
    for flow in root.findall('flow'):
        if 'departSpeed' in params:
            flow.set('departSpeed', params['departSpeed'])
        if 'departPos' in params:
            flow.set('departPos', params['departPos'])
        if 'departLane' in params:
            # Only change numerical lanes. Don't overwrite random/free if already set.
            current_lane = flow.get('departLane', '0')
            if current_lane.isdigit() and params['departLane'] == 'best':
                 flow.set('departLane', 'best')
            elif params['departLane'] != 'best':
                 # If we are testing '0', enforce the hardcoded lanes 
                 # (assumes original XML has L0 -> 0, L1 -> 1, L2 -> 2)
                 pass # keep original
                 
    tree.write(sweep_file)
    return sweep_file

def evaluate_params(scenario: str, params: dict, num_steps: int = 7200) -> dict:
    """Evaluate a single parameter combination for a shorter duration (e.g. 1 hour)."""
    # 1. Create temporary XML with new params
    sweep_xml_path = create_sweep_xml(scenario, params)
    
    # 2. Update SUMO Config to Use the Sweep XML
    scenario_dir = os.path.join(DATA_SOURCE_DIR, scenario)
    sumocfg_path = os.path.join(scenario_dir, 'Configuration.sumocfg')
    
    # Read the sumocfg and temporarily rewrite the route-files tag
    tree = ET.parse(sumocfg_path)
    root = tree.getroot()
    input_tag = root.find('input')
    route_tag = input_tag.find('route-files')
    orig_route = route_tag.get('value')
    route_tag.set('value', 'Configuration_sweep.rou.xml')
    
    sweep_sumocfg_path = os.path.join(scenario_dir, 'Configuration_sweep.sumocfg')
    tree.write(sweep_sumocfg_path)
    
    # Keep track of flows
    vehicle_flows = []
    
    try:
        # We start SUMO manually here (bypassing the env's standard start) so we can 
        # point it to the sweep sumocfg and run for a shorter time to save evaluation time.
        # But to keep things simple and comparable, we just use the existing Env with 
        # a slight monkey-patch to self.sumo_cfg
        env = SumoRampMeteringEnv(scenario=scenario, use_gui=False, add_noise=False)
        env.sumo_cfg = sweep_sumocfg_path # Overwrite config
        controller = NoControlController()
        
        # We only run for 'num_steps' simulation seconds instead of the full 7200.
        # So we need to do this manually instead of using env.step() loop which is fixed length.
        # Better yet, since env.step() advances by 60 seconds (delta_time), we can just step it N times.
        
        control_periods = num_steps // env.delta_time
        
        observation, _ = env.reset(seed=RANDOM_SEED)
        
        for _ in range(control_periods):
            action = controller.get_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
                
        stats = env.get_episode_statistics()
        vehicle_flows = stats.get('vehicle_flows', [])
        avg_speed = stats.get('avg_speed_kmh', 0)
        env.close()
        
    finally:
        # Cleanup
        if os.path.exists(sweep_xml_path):
            os.remove(sweep_xml_path)
        if os.path.exists(sweep_sumocfg_path):
            os.remove(sweep_sumocfg_path)

    # 3. Analyze flows
    theory_main, theory_ramp = parse_xml_demand(scenario_dir)
    
    # Aggregate actuals
    times = [int(t // 60) for t, edge, vtype in vehicle_flows if t <= num_steps]
    edges = [edge for t, edge, vtype in vehicle_flows if t <= num_steps]
    
    actual_mainline = sum(1 for e in edges if e in {'E_main_before', 'E_upstream'})
    actual_ramp = sum(1 for e in edges if e == 'E_ramp')
    
    # Sum theory for the equivalent time period
    max_min = num_steps // 60
    theory_main_sum = sum(theory_main[:max_min]) / 60.0 # Convert veh/h back to veh/min sum
    theory_ramp_sum = sum(theory_ramp[:max_min]) / 60.0
    
    # Metrics
    mainline_diff = actual_mainline - theory_main_sum
    ramp_diff = actual_ramp - theory_ramp_sum
    total_abs_error = abs(mainline_diff) + abs(ramp_diff)
    
    result = deepcopy(params)
    result.update({
        'actual_mainline': actual_mainline,
        'theory_mainline': theory_main_sum,
        'actual_ramp': actual_ramp,
        'theory_ramp': theory_ramp_sum,
        'total_abs_error': total_abs_error,
        'avg_speed': avg_speed
    })
    
    return result

def main():
    scenario = "stationary"
    print(f"Sweeping demand insertion parameters for scenario: {scenario}")
    
    # Generate all combinations
    keys = PARAM_GRID.keys()
    values = PARAM_GRID.values()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Total combinations to evaluate: {len(combinations)}")
    
    results = []
    
    # Run 1 hour (3600s) to evaluate injection rate speed
    eval_seconds = 3600 
    
    for params in tqdm(combinations, desc="Evaluating combinations"):
        res = evaluate_params(scenario, params, num_steps=eval_seconds)
        results.append(res)
        
    df = pd.DataFrame(results)
    
    # Sort by error (lowest is best)
    df = df.sort_values('total_abs_error')
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / 'demand_parameter_sweep.csv'
    df.to_csv(out_path, index=False)
    
    print(f"\nSaved results to: {out_path}")
    print("\nBest 5 configurations:")
    print(df.head(5).to_string())

if __name__ == "__main__":
    main()
