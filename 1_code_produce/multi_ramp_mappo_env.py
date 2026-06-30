"""
VecEnv implementation for Multi-Agent PPO (MAPPO) on the 2-ramp scenario.
This maps exactly to the paper's Decentralized architecture with Shared Reward:
- Each agent observes its own ramp's occupancies + the other agent's reward. (5 dims)
- Each agent outputs a scalar metering rate for its own ramp. (1 dim)
- Stable Baselines 3 interacts with this as if there were 2 independent parallel environments,
  therefore training a shared policy network over the experiences of both agents simultaneously. 
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Sequence

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import traci
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvIndices

from config import (
    CONTROL_STEPS,
    SIM_STEP,
    NUM_CONTROL_PERIODS,
    GREEN_PHASE,
    MIN_RED,
    MAX_RED,
    MR_MIN,
    MR_MAX,
    ETA,
    SUMO_BINARY,
)

DATA_DIR = Path(__file__).parent.parent / "1_data_source" / "sumo_simulation_multi_ramp" / "flat_peak"
SUMO_CFG = str(DATA_DIR / "Configuration.sumocfg")
TLS_IDS = ["ramp_meter", "ramp_meter2"]
QUEUE_IDS = ["queue_ramp_1", "queue_ramp_2"]

OBS_DETECTORS = {
    0: {
        "p2": ["det_r1_p2_l0", "det_r1_p2_l1", "det_r1_p2_l2"],
        "p3": ["det_r1_p3_l0", "det_r1_p3_l1", "det_r1_p3_l2"],
        "p4": ["det_r1_p4_l1", "det_r1_p4_l2", "det_r1_p4_l3"],
        "p5": ["det_r1_p5_l0", "det_r1_p5_l1", "det_r1_p5_l2"],
    },
    1: {
        "p2": ["det_r2_p2_l0", "det_r2_p2_l1", "det_r2_p2_l2"],
        "p3": ["det_r2_p3_l0", "det_r2_p3_l1", "det_r2_p3_l2"],
        "p4": ["det_r2_p4_l1", "det_r2_p4_l2", "det_r2_p4_l3"],
        "p5": ["det_r2_p5_l0", "det_r2_p5_l1", "det_r2_p5_l2"],
    },
}

P4_MAINLINE = {
    0: ["det_r1_p4_l1", "det_r1_p4_l2", "det_r1_p4_l3"],
    1: ["det_r2_p4_l1", "det_r2_p4_l2", "det_r2_p4_l3"],
}


class MultiRampMAPPOVecEnv(VecEnv):
    """
    A custom Vectorized Environment that transparently runs 2 agents in a 
    single SUMO simulation for MAPPO training.
    """

    def __init__(self, use_gui: bool = False, seed: int = 42):
        self.num_agents = 2
        # 5 dims: 4 occupancies [0,1] + 1 normalized neighbor reward [0,1]
        obs_space = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)
        act_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        
        super().__init__(num_envs=self.num_agents, observation_space=obs_space, action_space=act_space)
        
        self.use_gui = use_gui
        self.sumo_cfg = SUMO_CFG
        self.env_seed = seed
        
        self._conn_label = f"mappo_vec_env_{id(self)}"
        self._sumo_running = False
        self.sumo_binary = self._find_sumo_binary(use_gui)

        self.current_step = 0
        self.actions = np.zeros((self.num_agents, 1), dtype=np.float32)
        self.last_rewards = np.zeros(self.num_agents, dtype=np.float32)
        self.episode_returns = np.zeros(self.num_agents, dtype=np.float32)

    def _find_sumo_binary(self, use_gui: bool) -> str:
        if use_gui:
            sumo_dir = os.path.dirname(SUMO_BINARY)
            gui = os.path.join(sumo_dir, "sumo-gui")
            if os.path.exists(gui + ".exe"):
                return gui + ".exe"
            return gui if os.path.exists(gui) else "sumo-gui"
        return SUMO_BINARY

    def _start_sumo(self):
        cmd = [
            self.sumo_binary,
            "-c", self.sumo_cfg,
            "--start",
            "--quit-on-end",
            "--no-warnings",
            "--no-step-log",
            "--step-length", str(SIM_STEP),
            "--seed", str(self.env_seed)
        ]
        traci.start(cmd, label=self._conn_label)
        self._sumo_running = True

    def _close_sumo(self):
        if self._sumo_running:
            try:
                traci.switch(self._conn_label)
                traci.close()
            except traci.exceptions.FatalTraCIError:
                pass
            self._sumo_running = False

    def reset(self) -> np.ndarray:
        self._close_sumo()
        self._start_sumo()
        traci.switch(self._conn_label)

        self.current_step = 0
        self.last_rewards = np.zeros(self.num_agents, dtype=np.float32)
        self.episode_returns = np.zeros(self.num_agents, dtype=np.float32)

        for _ in range(CONTROL_STEPS):
            traci.simulationStep()

        return self._get_obs()

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        traci.switch(self._conn_label)
        
        # Apply actions
        a0 = float(np.clip(self.actions[0, 0], 0.0, 1.0))
        a1 = float(np.clip(self.actions[1, 0], 0.0, 1.0))
        self._apply_metering_rate(TLS_IDS[0], a0)
        self._apply_metering_rate(TLS_IDS[1], a1)

        # Step simulation
        for _ in range(CONTROL_STEPS):
            traci.simulationStep()

        self.current_step += 1
        done = self.current_step >= NUM_CONTROL_PERIODS

        # Gather speeds and queues
        s1 = self._read_mean_speed_kmh(P4_MAINLINE[0])
        s2 = self._read_mean_speed_kmh(P4_MAINLINE[1])
        q1 = self._read_queue_m(QUEUE_IDS[0])
        q2 = self._read_queue_m(QUEUE_IDS[1])

        # Compute rewards
        r1 = s1 - ETA * q1
        r2 = s2 - ETA * q2
        self.last_rewards = np.array([r1, r2], dtype=np.float32)
        self.episode_returns += self.last_rewards

        # Gather obs
        obs = self._get_obs()
        dones = np.array([done, done], dtype=bool)

        infos = []
        for i in range(self.num_agents):
            info = {
                "step": self.current_step,
                "reward": float(self.last_rewards[i]),
                "speed": s1 if i == 0 else s2,
                "queue": q1 if i == 0 else q2,
            }
            if done:
                # SB3 standard dictates that terminal observation goes in info
                info["terminal_observation"] = obs[i].copy()
                info["episode"] = {"r": float(self.episode_returns[i]), "l": self.current_step}
            infos.append(info)

        if done:
            # If done, returning obs should be from the newly reset environment
            obs = self.reset()

        return obs, self.last_rewards, dones, infos

    def _get_obs(self) -> np.ndarray:
        # Paper MAPPO state definition: 4 profile occupancies + neighbor's reward
        # Neighbor reward is normalized to [0,1] by dividing by 100 and clipping,
        # so all 5 features are on a comparable scale for the policy network.
        obs_all = []
        for agent_idx in range(self.num_agents):
            d = OBS_DETECTORS[agent_idx]
            occ_p2 = self._read_mean_occupancy(d["p2"])
            occ_p3 = self._read_mean_occupancy(d["p3"])
            occ_p4 = self._read_mean_occupancy(d["p4"])
            occ_p5 = self._read_mean_occupancy(d["p5"])
            # Normalize neighbor reward: typical range ~0-80 km/h → /100 → ~0-0.8
            neighbor_reward_norm = float(np.clip(self.last_rewards[1 - agent_idx] / 100.0, 0.0, 1.0))
            
            obs_all.append([occ_p2, occ_p3, occ_p4, occ_p5, neighbor_reward_norm])
            
        return np.array(obs_all, dtype=np.float32)

    def _read_mean_occupancy(self, detector_ids) -> float:
        vals = []
        for det_id in detector_ids:
            try:
                vals.append(traci.inductionloop.getLastIntervalOccupancy(det_id) / 100.0)
            except:
                pass
        return float(np.mean(vals)) if vals else 0.0

    def _read_mean_speed_kmh(self, detector_ids) -> float:
        vals = []
        for det_id in detector_ids:
            try:
                v = traci.inductionloop.getLastIntervalMeanSpeed(det_id)
                if v >= 0:
                    vals.append(v * 3.6)
            except:
                pass
        return float(np.mean(vals)) if vals else 0.0

    def _read_queue_m(self, queue_id: str) -> float:
        try:
            return float(traci.lanearea.getJamLengthMeters(queue_id))
        except:
            return 0.0

    def _metering_rate_to_red_duration(self, metering_rate: float) -> float:
        if metering_rate <= 0:
            return MAX_RED
        cycle_time = 3600.0 / metering_rate
        red_duration = cycle_time - GREEN_PHASE
        return float(np.clip(red_duration, MIN_RED, MAX_RED))

    def _apply_metering_rate(self, tls_id: str, action: float):
        metering_rate = MR_MIN + action * (MR_MAX - MR_MIN)
        red_duration = self._metering_rate_to_red_duration(metering_rate)

        try:
            n_links = len(traci.trafficlight.getControlledLinks(tls_id))
        except:
            n_links = 4
        if n_links < 1:
            return

        state_green = "G" * n_links
        state_red = "r" + "G" * (n_links - 1)

        logic = traci.trafficlight.Logic(
            programID="mappo",
            type=0,                  
            currentPhaseIndex=0,
            phases=[
                traci.trafficlight.Phase(duration=GREEN_PHASE, state=state_green),
                traci.trafficlight.Phase(duration=red_duration, state=state_red),
            ],
        )
        traci.trafficlight.setProgramLogic(tls_id, logic)

    def close(self):
        self._close_sumo()

    # Stub methods required by VecEnv
    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        # Return dummy values if SB3 asks for internal env attributes
        return [None] * self.num_envs
    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        pass
    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        return [None] * self.num_envs
    def env_is_wrapped(self, wrapper_class: type, indices: VecEnvIndices = None) -> List[bool]:
        return [False] * self.num_envs
