"""
Quick diagnostic: run 1 episode of ALINEA and log p4-mainline occupancy,
action, metering rate, and queue at every control step.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import traci
from sumo_env import SumoRampMeteringEnv
from controllers.alinea import AlineaController, ALINEA_P4_MAINLINE_DETS
from config import MR_MIN, MR_MAX, DETECTOR_CONFIG, get_detector_ids

env = SumoRampMeteringEnv(scenario="stationary", use_gui=False)
ctrl = AlineaController()  # default O_crit=0.16, KR=0.35

obs, _ = env.reset()
ctrl.reset()
done = False

print(f"{'step':>4}  {'occ_p4ml':>8}  {'occ_p5':>8}  {'action':>7}  {'MR':>7}  {'queue':>7}")
print("-" * 60)

step = 0
while not done:
    action = ctrl.get_action(obs)
    mr = MR_MIN + action[0] * (MR_MAX - MR_MIN)

    # Read p4 mainline occ (same as ALINEA reads)
    traci.switch(env._conn_label)
    occs_p4 = []
    for d in ALINEA_P4_MAINLINE_DETS:
        try:
            occs_p4.append(traci.inductionloop.getLastIntervalOccupancy(d) / 100.0)
        except:
            pass
    occ_p4 = np.mean(occs_p4) if occs_p4 else 0.0

    # Also read p5 for comparison
    occs_p5 = []
    for d in get_detector_ids('p5'):
        try:
            occs_p5.append(traci.inductionloop.getLastIntervalOccupancy(d) / 100.0)
        except:
            pass
    occ_p5 = np.mean(occs_p5) if occs_p5 else 0.0

    queue = traci.lanearea.getJamLengthMeters("queue_ramp")

    print(f"{step:4d}  {occ_p4:8.4f}  {occ_p5:8.4f}  {action[0]:7.4f}  {mr:7.1f}  {queue:7.1f}")

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    ctrl.step()
    step += 1

env.close()
stats = env.get_episode_statistics()
print(f"\nAvg speed: {stats['avg_speed_kmh']:.2f} km/h")
print(f"Avg queue: {stats['avg_queue_m']:.2f} m")
print(f"Return:    {stats['total_return']:.2f}")
