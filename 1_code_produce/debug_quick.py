"""Quick debug: run one episode and print per-step flow/speed data with departure counts."""
import os, sys
os.environ['SUMO_HOME'] = r'C:\Users\mprosperi\Desktop\sumo-1.26.0'
sys.path.insert(0, os.path.join(os.environ['SUMO_HOME'], 'tools'))

import traci
from config import *

sumocfg = get_sumo_config("stationary")

traci.start([SUMO_BINARY, '-c', sumocfg, '--no-step-log'])

# Run for 2 hours (7200 s) with 60s control period
total_departed = 0
total_arrived = 0
total_teleported = 0

print(f"{'Time':>6} {'MainFlow':>10} {'RampFlow':>10} {'AfterFlow':>10} {'MergeSpd':>10} {'AfterSpd':>10} {'Queue':>8} {'Depart':>8} {'Active':>8}")

for t in range(0, 7200, 60):
    period_departed = 0
    for _ in range(120):  # 60s / 0.5s step-length = 120 steps
        traci.simulationStep()
        total_departed += traci.simulation.getDepartedNumber()
        total_arrived += traci.simulation.getArrivedNumber()
        period_departed += traci.simulation.getDepartedNumber()

    # Main flow at p3 (3 lanes)
    main_count = 0
    for lane in range(3):
        det_id = f'det_main_p3_l{lane}'
        try:
            main_count += traci.inductionloop.getLastIntervalVehicleNumber(det_id)
        except:
            pass
    
    # Ramp flow
    ramp_count = 0
    try:
        ramp_count = traci.inductionloop.getLastIntervalVehicleNumber('det_main_p4_l0_acceleration_lane')
    except:
        pass
    
    # After flow at p5 (2 lanes)
    after_count = 0
    for lane in range(2):
        det_id = f'det_main_p5_l{lane}'
        try:
            after_count += traci.inductionloop.getLastIntervalVehicleNumber(det_id)
        except:
            pass
    
    # Merge speed (p4, 4 lanes)
    merge_speeds = []
    for lane in range(4):
        det_id = f'det_main_p4_l{lane}' if lane > 0 else 'det_main_p4_l0_acceleration_lane'
        try:
            s = traci.inductionloop.getLastIntervalMeanSpeed(det_id)
            if s >= 0:
                merge_speeds.append(s * 3.6)
        except:
            pass
    
    # After speed (p5, 2 lanes)
    after_speeds = []
    for lane in range(2):
        det_id = f'det_main_p5_l{lane}'
        try:
            s = traci.inductionloop.getLastIntervalMeanSpeed(det_id)
            if s >= 0:
                after_speeds.append(s * 3.6)
        except:
            pass
    
    queue = 0
    try:
        queue = traci.lanearea.getJamLengthMeters('queue_ramp')
    except:
        pass
    
    merge_spd = sum(merge_speeds)/len(merge_speeds) if merge_speeds else 0
    after_spd = sum(after_speeds)/len(after_speeds) if after_speeds else 0
    
    main_flow = main_count * 60
    ramp_flow = ramp_count * 60  
    after_flow = after_count * 60
    active = traci.vehicle.getIDCount()
    
    print(f"{t+60:>6} {main_flow:>10} {ramp_flow:>10} {after_flow:>10} {merge_spd:>10.1f} {after_spd:>10.1f} {queue:>8.1f} {period_departed:>8} {active:>8}")

# Check TLS state
try:
    tls_state = traci.trafficlight.getRedYellowGreenState('ramp_meter')
    tls_prog = traci.trafficlight.getProgram('ramp_meter')
    print(f"\nTLS state: {tls_state}, program: {tls_prog}")
except Exception as e:
    print(f"\nCould not get TLS info: {e}")

# Vehicle counts
target_main = 6000 * 7200 / 3600  # total mainline vehicles expected
target_ramp = 1000 * 7200 / 3600  # total ramp vehicles expected
print(f"\nTotal departed: {total_departed} (target main: {target_main:.0f}, target ramp: {target_ramp:.0f}, total: {target_main+target_ramp:.0f})")
print(f"Total arrived: {total_arrived}")
print(f"Active vehicles at end: {traci.vehicle.getIDCount()}")
print(f"Waiting to depart: {traci.simulation.getMinExpectedNumber() - traci.vehicle.getIDCount()}")

traci.close()
