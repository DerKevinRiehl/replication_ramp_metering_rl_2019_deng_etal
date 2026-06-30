"""
Configuration file for 2019 Deng et al. replication study.
"Advanced Self-Improving Ramp Metering Algorithm based on Multi-Agent Deep Reinforcement Learning"

References from the paper are indicated with equation numbers (eq.X) or section numbers.
"""

import os
from pathlib import Path

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_SOURCE_DIR = BASE_DIR / "1_data_source" / "sumo_simulation_single_ramp"
DATA_PRODUCED_DIR = BASE_DIR / "2_data_produce"
MODELS_DIR = DATA_PRODUCED_DIR / "models"
LOGS_DIR = DATA_PRODUCED_DIR / "logs"
RESULTS_DIR = DATA_PRODUCED_DIR / "results"
VIS_CODE_DIR = BASE_DIR / "3_code_visualization"
VIS_DATA_DIR = BASE_DIR / "3_data_visualization"

# =============================================================================
# SIMULATION PARAMETERS (Section III.B)
# =============================================================================
SIMULATION_TIME = 7200  # 2 hours in seconds
CONTROL_PERIOD_SECONDS = 60     # 1 minute control period (Section III.B.1)
SIM_STEP = 0.5          # Simulation step size (0.5s for finer resolution/capacity)

# Number of simulation steps per control action (120 steps)
CONTROL_STEPS = int(CONTROL_PERIOD_SECONDS / SIM_STEP)

NUM_CONTROL_PERIODS = SIMULATION_TIME // CONTROL_PERIOD_SECONDS  # 120 periods per episode

# Ramp meter timing (Section II.A.2)
GREEN_PHASE = 3         # Green phase duration (seconds) - one-car-per-green
MIN_RED = 0             # Minimum red phase duration
MAX_RED = 57            # Maximum red phase duration (so cycle = 3+57 = 60s max)

# Metering rate bounds (derived from timing)
MR_MIN = 3600 / (GREEN_PHASE + MAX_RED)  # ~60 veh/h minimum
MR_MAX = 3600 / (GREEN_PHASE + MIN_RED)  # 1200 veh/h maximum (green only)

# =============================================================================
# STATE REPRESENTATION (Section II.A.1)
# =============================================================================
# Detector profiles for state observation (from Additional.add.xml)
# K = 4 profiles as recommended in paper: "K is at least 4 to depict the traffic state"
# Available profiles: p1 (pos 0), p2 (pos 200), p3 (pos 400), p4 (merge area), p5 (downstream)
DETECTOR_PROFILES = ['p2', 'p3', 'p4', 'p5']  # K = 4 profiles near ramp
DOWNSTREAM_PROFILE = 'p4'  # Bottleneck profile for reward calculation (p4 is in merge area)
NUM_STATE_FEATURES = len(DETECTOR_PROFILES)  # 4 occupancy values

# Number of lanes per detector profile (from your Network.net.xml)
NUM_MAINLINE_LANES = 3  # E_main_before has 3 lanes (l0, l1, l2)
NUM_RAMP_LANES = 1      # E_ramp has 1 lane

# Detector IDs mapping (exact IDs from Additional.add.xml)
# Profile p1-p3: on E_main_before (3 lanes each)
# Profile p4: on E_main_with_acc_lane (4 lanes - includes acceleration lane)
# Profile p5: on E_main_after (3 lanes — per paper figure, no lane drop)
DETECTOR_CONFIG = {
    'p1': {'lanes': ['det_main_p1_l0', 'det_main_p1_l1', 'det_main_p1_l2']},
    'p2': {'lanes': ['det_main_p2_l0', 'det_main_p2_l1', 'det_main_p2_l2']},
    'p3': {'lanes': ['det_main_p3_l0', 'det_main_p3_l1', 'det_main_p3_l2']},
    'p4': {'lanes': ['det_main_p4_l0_acceleration_lane', 'det_main_p4_l1', 'det_main_p4_l2', 'det_main_p4_l3']},
    'p5': {'lanes': ['det_main_p5_l0', 'det_main_p5_l1', 'det_main_p5_l2']},
}

# Queue detector ID (laneAreaDetector from Additional.add.xml)
QUEUE_DETECTOR_ID = "queue_ramp"

# =============================================================================
# REWARD FUNCTION (Section II.A.3, eq.3)
# =============================================================================
# r(t+1) = -v_k'(t) - η*q(t)
# where v_k' is speed at bottleneck, q is queue size
ETA = 0.1  # Coefficient for queue size penalty (η = 0.1 as stated in paper)

# =============================================================================
# ALINEA PARAMETERS (Section III.B.2)
# =============================================================================
# "We set the critical occupancy to 0.18 and the metering gain KR to 0.35"
# K_R operates in normalised action space [0, 1], NOT in veh/h.
# ALINEA formula (normalised):  a(t) = a(t-1) + K_R * (O_cr - O_down(t))
# The metering rate in veh/h is then: MR = MR_MIN + a * (MR_MAX - MR_MIN)
# Per-scenario calibrated values — update these after running sweep_ocrit.py
ALINEA_O_CRIT = {
    'stationary': 0.14,
    'flat_peak':  0.14,
    'sharp_peak': 0.14,
}
ALINEA_KR = {
    'stationary': 0.35,
    'flat_peak':  0.50,
    'sharp_peak': 0.20,
}
# Classical ALINEA measures downstream of the merge, so use p5 (after merge)
# rather than p4 (IN the merge area, includes acceleration lane which inflates occ).
ALINEA_DOWNSTREAM_PROFILE = 'p5'

# =============================================================================
# FIXED-TIME CONTROL PARAMETERS
# =============================================================================
# "Proper metering rates of fixed-time control are determined by enumeration"
# Default values - should be tuned per scenario
FIXED_TIME_RATES = {
    'stationary': 515,      # tau=1.5 grid search best (paper_distance=0.913): speed=57.8, queue=11.7
    'flat_peak': 515,       # tau=1.5 grid search best (paper_distance=0.762): speed=57.1, queue=12.9
    'sharp_peak': 515,      # tau=1.5 grid search best (paper_distance=0.696): speed=56.6, queue=13.7
}

# =============================================================================
# PPO HYPERPARAMETERS (Section II.C)
# =============================================================================
# Neural network architecture (Figure 2 — separate actor and critic networks)
PPO_POLICY = "MlpPolicy"
PPO_NET_ARCH = dict(pi=[64, 64], vf=[64, 64])  # Separate actor-critic (paper Fig. 2)

# Policy initialisation
# Default log_std_init=0.0 → std=1.0 which is far too wide for action space [0,1].
# With std=1.0 most samples hit the clip boundaries, making actions essentially random.
# std ≈ 0.37 (log_std_init=-1.0) gives 95% of samples within μ±0.74 — enough
# exploration while still being informative for the gradient.
PPO_LOG_STD_INIT = -1.0  # Initial policy std ≈ 0.37

# Training parameters
PPO_LEARNING_RATE = 3e-2
PPO_N_STEPS = 2048       # Steps per update
PPO_BATCH_SIZE = 64
PPO_N_EPOCHS = 10        # Epochs per update
PPO_GAMMA = 0.99         # Discount factor
PPO_GAE_LAMBDA = 0.95    # GAE lambda
PPO_CLIP_RANGE = 0.2     # PPO clip range
PPO_ENT_COEF = 0.01      # Small entropy bonus to encourage exploration
PPO_VF_COEF = 0.5        # Value function coefficient
PPO_MAX_GRAD_NORM = 0.5  # Gradient clipping

# Training episodes (Section III.C.1)
TRAINING_EPISODES = 1000  # "Training process ... when the episode number reached 1000"
TRAINING_TIMESTEPS = TRAINING_EPISODES * NUM_CONTROL_PERIODS

# =============================================================================
# DEMAND SCENARIOS (Section III.B.1, Figure 5)
# =============================================================================
SCENARIOS = ['stationary', 'flat_peak', 'sharp_peak']

# Demand profiles as piecewise-constant intervals (begin_s, end_s, veh/h).
# These are extracted directly from Configuration.rou.xml for each scenario,
# so the config is always aligned with what SUMO actually simulates.

MAINLINE_FLOW = {
    'stationary': [
        (0, 7200, 6000.00),
    ],
    'flat_peak': [
        (0,    600,  5390.00),
        (600,  1200, 5940.00),
        (1200, 1800, 6490.00),
        (1800, 2400, 7040.00),
        (2400, 3000, 7370.00),
        (3000, 4200, 7700.00),
        (4200, 4800, 7370.00),
        (4800, 5400, 7040.00),
        (5400, 6000, 6490.00),
        (6000, 6600, 5940.00),
        (6600, 7200, 5390.00),
    ],
    'sharp_peak': [
        (0,    600,  3750.00),
        (600,  1200, 3829.79),
        (1200, 1800, 4137.93),
        (1800, 2400, 4736.84),
        (2400, 3000, 5714.29),
        (3000, 3600, 6666.67),
        (3600, 4200, 7058.82),
        (4200, 4800, 6666.67),
        (4800, 5400, 5714.29),
        (5400, 6000, 4736.84),
        (6000, 6600, 4137.93),
        (6600, 7200, 3829.79),
    ],
}

RAMP_FLOW = {
    'stationary': [
        (0, 7200, 1000.00),
    ],
    'flat_peak': [
        (0,    600,   990.00),
        (600,  1200, 1155.00),
        (1200, 1800, 1320.00),
        (1800, 2400, 1485.00),
        (2400, 4800, 1550.00),
        (4800, 5400, 1485.00),
        (5400, 6000, 1320.00),
        (6000, 6600, 1155.00),
        (6600, 7200,  990.00),
    ],
    'sharp_peak': [
        (0,    600,   800.00),
        (600,  1200,  823.80),
        (1200, 1800,  888.89),
        (1800, 2400, 1022.73),
        (2400, 3000, 1220.34),
        (3000, 3600, 1417.32),
        (3600, 4200, 1500.00),
        (4200, 4800, 1417.32),
        (4800, 5400, 1220.34),
        (5400, 6000, 1022.73),
        (6000, 6600,  888.89),
        (6600, 7200,  823.80),
    ],
}

# =============================================================================
# SUMO CONFIGURATION
# =============================================================================
# Use "sumo-gui" to visualize the simulation, "sumo" for headless
# Set via environment variable SUMO_BINARY or change default here

SUMO_BINARY = os.getenv("SUMO_BINARY", r"C:\Users\mprosperi\Desktop\sumo-1.26.0\bin\sumo.exe")
SUMO_CONFIG_FILE = "Configuration.sumocfg"

# Traffic light ID for ramp meter (from Network.net.xml)
TLS_ID = "ramp_meter"

# Edge IDs (from Network.net.xml)
EDGE_MAIN_BEFORE = "E_main_before"
EDGE_MAIN_WITH_ACC = "E_main_with_acc_lane"
EDGE_MAIN_AFTER = "E_main_after"
EDGE_RAMP = "E_ramp"

# =============================================================================
# EVALUATION SETTINGS
# =============================================================================
EVAL_EPISODES = 10  # Number of evaluation episodes per scenario
RANDOM_SEED = 42

# =============================================================================
# CONTROLLER TYPES
# =============================================================================
CONTROLLERS = ['no_control', 'fixed_time', 'alinea', 'ppo']

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_scenario_path(scenario: str) -> Path:
    """Get the path to scenario SUMO files."""
    return DATA_SOURCE_DIR / scenario

def get_sumo_config(scenario: str) -> str:
    """Get full path to SUMO configuration file."""
    return str(get_scenario_path(scenario) / SUMO_CONFIG_FILE)

def get_model_path(scenario: str, episodes: int = TRAINING_EPISODES) -> Path:
    """Get path for saving/loading trained model."""
    return MODELS_DIR / f"ppo_{scenario}_{episodes}ep.zip"

def get_results_path(scenario: str) -> Path:
    """Get path for saving evaluation results."""
    return RESULTS_DIR / f"{scenario}_results.csv"

def get_detector_ids(profile: str) -> list:
    """Get list of detector IDs for a profile."""
    return DETECTOR_CONFIG.get(profile, {}).get('lanes', [])

def ensure_dirs():
    """Create output directories if they don't exist."""
    for d in [MODELS_DIR, LOGS_DIR / "tensorboard", RESULTS_DIR, VIS_DATA_DIR]:
        d.mkdir(parents=True, exist_ok=True)

# Initialize directories on import
ensure_dirs()
