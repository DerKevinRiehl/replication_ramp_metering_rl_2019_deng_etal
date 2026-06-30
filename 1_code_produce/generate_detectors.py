"""
Generate Additional.add.xml with 30 evenly-spaced detector profiles
along the mainline for space-time visualization, plus the original
p2-p5 detectors, e1, and e2 queue detectors for BOTH single and multi-ramp.
"""

import os

# --- Single Ramp Definitions ---
EDGES_SINGLE = [
    {'id': 'E_main_before',       'length': 404.42, 'lanes': 3, 'lane_prefix': 'E_main_before'},
    {'id': 'E_main_with_acc_lane','length': 122.59, 'lanes': 4, 'lane_prefix': 'E_main_with_acc_lane'},
    {'id': 'E_main_after',        'length': 461.00, 'lanes': 3, 'lane_prefix': 'E_main_after'},
]
TOTAL_LENGTH_SINGLE = sum(e['length'] for e in EDGES_SINGLE)

ORIGINAL_DETECTORS_SINGLE = [
    # p1: E_main_before, pos=0
    {'id': 'det_main_p1_l0', 'lane': 'E_main_before_0', 'pos': '0.00', 'file': 'detectors_p1.xml', 'extra': 'friendlyPos="1"'},
    {'id': 'det_main_p1_l1', 'lane': 'E_main_before_1', 'pos': '0.00', 'file': 'detectors_p1.xml', 'extra': 'friendlyPos="1"'},
    {'id': 'det_main_p1_l2', 'lane': 'E_main_before_2', 'pos': '0.00', 'file': 'detectors_p1.xml', 'extra': 'friendlyPos="1"'},
    # p2: E_main_before, pos=200
    {'id': 'det_main_p2_l0', 'lane': 'E_main_before_0', 'pos': '200.00', 'file': 'detectors_p2.xml', 'extra': 'friendlyPos="1"'},
    {'id': 'det_main_p2_l1', 'lane': 'E_main_before_1', 'pos': '200.00', 'file': 'detectors_p2.xml', 'extra': 'friendlyPos="1"'},
    {'id': 'det_main_p2_l2', 'lane': 'E_main_before_2', 'pos': '200.00', 'file': 'detectors_p2.xml', 'extra': 'friendlyPos="1"'},
    # p3: E_main_before, pos=400
    {'id': 'det_main_p3_l0', 'lane': 'E_main_before_0', 'pos': '400.00', 'file': 'detectors_p3.xml', 'extra': 'friendlyPos="1"'},
    {'id': 'det_main_p3_l1', 'lane': 'E_main_before_1', 'pos': '400.00', 'file': 'detectors_p3.xml', 'extra': 'friendlyPos="1"'},
    {'id': 'det_main_p3_l2', 'lane': 'E_main_before_2', 'pos': '400.00', 'file': 'detectors_p3.xml', 'extra': 'friendlyPos="1"'},
    # p4: E_main_with_acc_lane, pos=70
    {'id': 'det_main_p4_l0_acceleration_lane', 'lane': 'E_main_with_acc_lane_0', 'pos': '70.00', 'file': 'detectors_p4.xml', 'extra': 'friendlyPos="1"'},
    {'id': 'det_main_p4_l1', 'lane': 'E_main_with_acc_lane_1', 'pos': '70.00', 'file': 'detectors_p4.xml', 'extra': 'friendlyPos="1"'},
    {'id': 'det_main_p4_l2', 'lane': 'E_main_with_acc_lane_2', 'pos': '70.00', 'file': 'detectors_p4.xml', 'extra': 'friendlyPos="1"'},
    {'id': 'det_main_p4_l3', 'lane': 'E_main_with_acc_lane_3', 'pos': '70.00', 'file': 'detectors_p4.xml', 'extra': 'friendlyPos="1"'},
    # p5: E_main_after, pos=50
    {'id': 'det_main_p5_l0', 'lane': 'E_main_after_0', 'pos': '50.00', 'file': 'detectors_p5.xml', 'extra': 'friendlyPos="1"'},
    {'id': 'det_main_p5_l1', 'lane': 'E_main_after_1', 'pos': '50.00', 'file': 'detectors_p5.xml', 'extra': 'friendlyPos="1"'},
    {'id': 'det_main_p5_l2', 'lane': 'E_main_after_2', 'pos': '50.00', 'file': 'detectors_p5.xml', 'extra': 'friendlyPos="1"'},
]

# --- Multi Ramp Definitions ---
EDGES_MULTI = [
    {'id': 'E_main_before',          'length': 392.23, 'lanes': 3, 'lane_prefix': 'E_main_before'},
    {'id': 'E_main_with_acc_lane',   'length': 200.00, 'lanes': 4, 'lane_prefix': 'E_main_with_acc_lane'},
    {'id': 'E_main_after',           'length': 196.92, 'lanes': 3, 'lane_prefix': 'E_main_after'},
    {'id': 'E_main_with_acc_lane_2', 'length': 157.27, 'lanes': 4, 'lane_prefix': 'E_main_with_acc_lane_2'},
    {'id': 'E_main_after_2',         'length': 367.16, 'lanes': 3, 'lane_prefix': 'E_main_after_2'},
]
TOTAL_LENGTH_MULTI = sum(e['length'] for e in EDGES_MULTI)

ORIGINAL_DETECTORS_MULTI = [
    # Ramp 1
    {'id': 'det_r1_p2_l0', 'lane': 'E_main_before_0', 'pos': '200.00', 'file': 'detectors_r1_p2.xml'},
    {'id': 'det_r1_p2_l1', 'lane': 'E_main_before_1', 'pos': '200.00', 'file': 'detectors_r1_p2.xml'},
    {'id': 'det_r1_p2_l2', 'lane': 'E_main_before_2', 'pos': '200.00', 'file': 'detectors_r1_p2.xml'},

    {'id': 'det_r1_p3_l0', 'lane': 'E_main_before_0', 'pos': '382.71', 'file': 'detectors_r1_p3.xml', 'extra': 'friendlyPos="1"'},
    {'id': 'det_r1_p3_l1', 'lane': 'E_main_before_1', 'pos': '382.71', 'file': 'detectors_r1_p3.xml', 'extra': 'friendlyPos="1"'},
    {'id': 'det_r1_p3_l2', 'lane': 'E_main_before_2', 'pos': '382.71', 'file': 'detectors_r1_p3.xml', 'extra': 'friendlyPos="1"'},

    {'id': 'det_r1_p4_l1', 'lane': 'E_main_with_acc_lane_1', 'pos': '70.00', 'file': 'detectors_r1_p4.xml'},
    {'id': 'det_r1_p4_l2', 'lane': 'E_main_with_acc_lane_2', 'pos': '70.00', 'file': 'detectors_r1_p4.xml'},
    {'id': 'det_r1_p4_l3', 'lane': 'E_main_with_acc_lane_3', 'pos': '70.00', 'file': 'detectors_r1_p4.xml'},

    {'id': 'det_r1_p5_l0', 'lane': 'E_main_after_0', 'pos': '50.00', 'file': 'detectors_r1_p5.xml'},
    {'id': 'det_r1_p5_l1', 'lane': 'E_main_after_1', 'pos': '50.00', 'file': 'detectors_r1_p5.xml'},
    {'id': 'det_r1_p5_l2', 'lane': 'E_main_after_2', 'pos': '50.00', 'file': 'detectors_r1_p5.xml'},

    # Ramp 2
    {'id': 'det_r2_p2_l0', 'lane': 'E_main_after_0', 'pos': '100.00', 'file': 'detectors_r2_p2.xml'},
    {'id': 'det_r2_p2_l1', 'lane': 'E_main_after_1', 'pos': '100.00', 'file': 'detectors_r2_p2.xml'},
    {'id': 'det_r2_p2_l2', 'lane': 'E_main_after_2', 'pos': '100.00', 'file': 'detectors_r2_p2.xml'},

    {'id': 'det_r2_p3_l0', 'lane': 'E_main_after_0', 'pos': '190.00', 'file': 'detectors_r2_p3.xml', 'extra': 'friendlyPos="1"'},
    {'id': 'det_r2_p3_l1', 'lane': 'E_main_after_1', 'pos': '190.00', 'file': 'detectors_r2_p3.xml', 'extra': 'friendlyPos="1"'},
    {'id': 'det_r2_p3_l2', 'lane': 'E_main_after_2', 'pos': '190.00', 'file': 'detectors_r2_p3.xml', 'extra': 'friendlyPos="1"'},

    {'id': 'det_r2_p4_l1', 'lane': 'E_main_with_acc_lane_2_1', 'pos': '70.00', 'file': 'detectors_r2_p4.xml'},
    {'id': 'det_r2_p4_l2', 'lane': 'E_main_with_acc_lane_2_2', 'pos': '70.00', 'file': 'detectors_r2_p4.xml'},
    {'id': 'det_r2_p4_l3', 'lane': 'E_main_with_acc_lane_2_3', 'pos': '70.00', 'file': 'detectors_r2_p4.xml'},

    {'id': 'det_r2_p5_l0', 'lane': 'E_main_after_2_0', 'pos': '50.00', 'file': 'detectors_r2_p5.xml'},
    {'id': 'det_r2_p5_l1', 'lane': 'E_main_after_2_1', 'pos': '50.00', 'file': 'detectors_r2_p5.xml'},
    {'id': 'det_r2_p5_l2', 'lane': 'E_main_after_2_2', 'pos': '50.00', 'file': 'detectors_r2_p5.xml'},
]

N_PROFILES = 30


def global_pos_to_edge_pos(global_pos, edges):
    """Convert a global position (0 to TOTAL_LENGTH) to (edge, local_pos)."""
    cumulative = 0.0
    for edge in edges:
        if global_pos < cumulative + edge['length']:
            local_pos = global_pos - cumulative
            local_pos = max(1.0, min(local_pos, edge['length'] - 1.0))
            return edge, local_pos
        cumulative += edge['length']
    edge = edges[-1]
    return edge, edge['length'] - 1.0


def generate_additional_xml(output_path, is_multi=False):
    """Generate Additional.add.xml with visualization profiles + original + e1/e2 detectors."""
    edges = EDGES_MULTI if is_multi else EDGES_SINGLE
    total_length = TOTAL_LENGTH_MULTI if is_multi else TOTAL_LENGTH_SINGLE
    spacing = total_length / N_PROFILES
    rl_detectors = ORIGINAL_DETECTORS_MULTI if is_multi else ORIGINAL_DETECTORS_SINGLE
    
    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">')
    
    # RL Detectors
    lines.append('    <!-- Original detectors for RL observation -->')
    for det in rl_detectors:
        extra = f' {det["extra"]}' if "extra" in det else ""
        lines.append(f'    <inductionLoop id="{det["id"]}" lane="{det["lane"]}" pos="{det["pos"]}" period="60.00" file="{det["file"]}"{extra}/>')
    
    # Visualization Profiles
    lines.append('')
    lines.append('    <!-- Visualization detectors: 30 profiles for space-time diagrams -->')
    for i in range(N_PROFILES):
        global_pos = (i + 0.5) * spacing
        edge, local_pos = global_pos_to_edge_pos(global_pos, edges)
        profile_id = f'vis_{i+1:02d}'
        
        for lane_idx in range(edge['lanes']):
            det_id = f'det_{profile_id}_l{lane_idx}'
            lane_id = f'{edge["lane_prefix"]}_{lane_idx}' if '_' in edge["lane_prefix"] else f'{edge["lane_prefix"]}_{lane_idx}'
            lines.append(f'    <inductionLoop id="{det_id}" lane="{lane_id}" pos="{local_pos:.2f}" period="60.00" file="detectors_vis.xml"/>')
    
    # e1 and e2 detectors
    lines.append('')
    lines.append('    <!-- e1 and e2 detectors on ramps -->')
    if is_multi:
        lines.append('    <inductionLoop id="det_ramp1_e1" lane="E_ramp_0" pos="120.00" period="60.00" file="ramp1_e1.xml"/>')
        lines.append('    <inductionLoop id="det_ramp2_e1" lane="E_ramp2_0" pos="120.00" period="60.00" file="ramp2_e1.xml"/>')
        lines.append('    <laneAreaDetector id="queue_ramp_1" lane="E_ramp_0" pos="37.57" length="200.00" friendlyPos="1" period="300.00" file="queue_ramp_1.xml"/>')
        lines.append('    <laneAreaDetector id="queue_ramp_2" lane="E_ramp2_0" pos="37.57" length="200.00" friendlyPos="1" period="300.00" file="queue_ramp_2.xml"/>')
    else:
        # Generate e1 for single-ramp since it was requested
        lines.append('    <inductionLoop id="det_ramp_e1" lane="E_ramp_0" pos="120.00" period="60.00" file="ramp_e1.xml"/>')
        lines.append('    <laneAreaDetector id="queue_ramp" lane="E_ramp_0" pos="73.72" endPos="-1" period="60.00" file="queue_ramp.xml"/>')
    
    lines.append('</additional>')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\r\n'.join(lines) + '\r\n')
    
    print(f"Generated: {output_path} (Multi={is_multi})")


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    
    # Single ramp generated
    single_dir = os.path.join(base, '..', '1_data_source', 'sumo_simulation_single_ramp')
    for scenario in ['stationary', 'flat_peak', 'sharp_peak']:
        path = os.path.join(single_dir, scenario, 'Additional.add.xml')
        if os.path.exists(os.path.dirname(path)):
            generate_additional_xml(path, is_multi=False)
            
    # Multi ramp generated
    multi_dir = os.path.join(base, '..', '1_data_source', 'sumo_simulation_multi_ramp')
    for scenario in ['stationary', 'flat_peak', 'sharp_peak']:
        path = os.path.join(multi_dir, scenario, 'Additional.add.xml')
        # Only flat_peak exists right now
        if os.path.exists(os.path.dirname(path)):
            generate_additional_xml(path, is_multi=True)

if __name__ == '__main__':
    main()
