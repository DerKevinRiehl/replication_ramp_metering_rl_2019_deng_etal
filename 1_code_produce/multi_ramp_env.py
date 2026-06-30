"""
Gymnasium environment for the 2-ramp (multi-ramp) flat_peak scenario.

State: concatenated occupancies from both ramps (4 profiles each) => 8 dims
Action: two normalized metering rates [0,1], one per ramp meter
Reward: mean over ramps of (speed_at_p4 - ETA * queue)
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import traci

from config import (
    CONTROL_STEPS,
    SIM_STEP,
    SIMULATION_TIME,
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


class MultiRampMeteringEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, use_gui: bool = False):
        super().__init__()
        self.use_gui = use_gui
        self.sumo_cfg = SUMO_CFG

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        self._conn_label = f"multi_ramp_env_{id(self)}"
        self._sumo_running = False

        self.current_step = 0
        self.episode_reward = 0.0
        self.speed_r1 = []
        self.speed_r2 = []
        self.queue_r1 = []
        self.queue_r2 = []

        self.sumo_binary = self._find_sumo_binary(use_gui)

    def _find_sumo_binary(self, use_gui: bool) -> str:
        if use_gui:
            sumo_dir = os.path.dirname(SUMO_BINARY)
            gui = os.path.join(sumo_dir, "sumo-gui")
            if os.path.exists(gui + ".exe"):
                return gui + ".exe"
            if os.path.exists(gui):
                return gui
            return "sumo-gui"
        return SUMO_BINARY

    def _start_sumo(self, seed: Optional[int] = None):
        cmd = [
            self.sumo_binary,
            "-c", self.sumo_cfg,
            "--start",
            "--quit-on-end",
            "--no-warnings",
            "--no-step-log",
            "--step-length", str(SIM_STEP),
        ]
        if seed is not None:
            cmd.extend(["--seed", str(seed)])

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

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._close_sumo()
        self._start_sumo(seed=seed)
        traci.switch(self._conn_label)

        self.current_step = 0
        self.episode_reward = 0.0
        self.speed_r1 = []
        self.speed_r2 = []
        self.queue_r1 = []
        self.queue_r2 = []

        for _ in range(CONTROL_STEPS):
            traci.simulationStep()

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray):
        traci.switch(self._conn_label)
        action = np.array(action, dtype=np.float32).flatten()
        if action.shape[0] != 2:
            raise ValueError(f"Expected action shape (2,), got {action.shape}")

        self._apply_metering_rate(TLS_IDS[0], float(np.clip(action[0], 0.0, 1.0)))
        self._apply_metering_rate(TLS_IDS[1], float(np.clip(action[1], 0.0, 1.0)))

        for _ in range(CONTROL_STEPS):
            traci.simulationStep()

        s1 = self._read_mean_speed_kmh(P4_MAINLINE[0])
        s2 = self._read_mean_speed_kmh(P4_MAINLINE[1])
        q1 = self._read_queue_m(QUEUE_IDS[0])
        q2 = self._read_queue_m(QUEUE_IDS[1])

        r1 = s1 - ETA * q1
        r2 = s2 - ETA * q2
        reward = 0.5 * (r1 + r2)

        self.speed_r1.append(s1)
        self.speed_r2.append(s2)
        self.queue_r1.append(q1)
        self.queue_r2.append(q2)
        self.episode_reward += reward

        self.current_step += 1
        terminated = self.current_step >= NUM_CONTROL_PERIODS
        truncated = False

        obs = self._get_obs()
        info = self._get_info()
        info.update({
            "speed_r1": s1,
            "speed_r2": s2,
            "queue_r1": q1,
            "queue_r2": q2,
            "reward_r1": r1,
            "reward_r2": r2,
        })

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        obs = []
        for ramp in [0, 1]:
            d = OBS_DETECTORS[ramp]
            obs.extend([
                self._read_mean_occupancy(d["p2"]),
                self._read_mean_occupancy(d["p3"]),
                self._read_mean_occupancy(d["p4"]),
                self._read_mean_occupancy(d["p5"]),
            ])
        return np.array(obs, dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        return {
            "step": self.current_step,
            "sim_time": self.current_step * 60,
        }

    def _read_mean_occupancy(self, detector_ids) -> float:
        vals = []
        for det_id in detector_ids:
            try:
                vals.append(traci.inductionloop.getLastIntervalOccupancy(det_id) / 100.0)
            except traci.exceptions.TraCIException:
                pass
        return float(np.mean(vals)) if vals else 0.0

    def _read_mean_speed_kmh(self, detector_ids) -> float:
        vals = []
        for det_id in detector_ids:
            try:
                v = traci.inductionloop.getLastIntervalMeanSpeed(det_id)
                if v >= 0:
                    vals.append(v * 3.6)
            except traci.exceptions.TraCIException:
                pass
        return float(np.mean(vals)) if vals else 0.0

    def _read_queue_m(self, queue_id: str) -> float:
        try:
            return float(traci.lanearea.getJamLengthMeters(queue_id))
        except traci.exceptions.TraCIException:
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
        except traci.exceptions.TraCIException:
            n_links = 4

        if n_links < 1:
            return

        state_green = "G" * n_links
        state_red = "r" + "G" * (n_links - 1)

        logic = traci.trafficlight.Logic(
            programID="multi_train",
            type=0,
            currentPhaseIndex=0,
            phases=[
                traci.trafficlight.Phase(duration=GREEN_PHASE, state=state_green),
                traci.trafficlight.Phase(duration=red_duration, state=state_red),
            ],
        )
        traci.trafficlight.setProgramLogic(tls_id, logic)

    def get_episode_statistics(self) -> Dict[str, float]:
        return {
            "avg_speed_r1_kmh": float(np.mean(self.speed_r1)) if self.speed_r1 else 0.0,
            "avg_speed_r2_kmh": float(np.mean(self.speed_r2)) if self.speed_r2 else 0.0,
            "avg_queue_r1_m": float(np.mean(self.queue_r1)) if self.queue_r1 else 0.0,
            "avg_queue_r2_m": float(np.mean(self.queue_r2)) if self.queue_r2 else 0.0,
            "avg_speed_kmh": float(np.mean(self.speed_r1 + self.speed_r2)) if (self.speed_r1 or self.speed_r2) else 0.0,
            "avg_queue_m": float(np.mean(self.queue_r1 + self.queue_r2)) if (self.queue_r1 or self.queue_r2) else 0.0,
            "total_return": float(self.episode_reward),
        }

    def close(self):
        self._close_sumo()
