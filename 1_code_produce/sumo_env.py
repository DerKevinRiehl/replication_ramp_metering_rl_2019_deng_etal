"""
SUMO Gymnasium Environment for Ramp Metering.
Implements the simulation interface described in Section III.A of Deng et al. (2019).

This environment wraps SUMO to provide a Gymnasium-compatible interface for
training reinforcement learning agents on the ramp metering task.
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any
import traci
import sumolib

from config import (
    CONTROL_STEPS, CONTROL_PERIOD_SECONDS, SIM_STEP, SIMULATION_TIME, NUM_CONTROL_PERIODS,
    GREEN_PHASE, MIN_RED, MAX_RED, MR_MIN, MR_MAX,
    DETECTOR_PROFILES, DOWNSTREAM_PROFILE, DETECTOR_CONFIG,
    ETA, TLS_ID, EDGE_RAMP, QUEUE_DETECTOR_ID,
    get_sumo_config, get_detector_ids, SUMO_BINARY
)


class SumoRampMeteringEnv(gym.Env):
    """
    Gymnasium environment for ramp metering using SUMO.
    
    State: Occupancy values from K detector profiles (K=4 by default)
    Action: Metering rate (continuous, normalized to [0, 1])
    Reward: -speed_bottleneck - eta * queue_size (eq. 3 from paper)
    
    The environment follows the single-ramp scenario from Section III.B.1
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        scenario: str = "stationary",
        render_mode: Optional[str] = None,
        use_gui: bool = False,
        delta_time: int = CONTROL_STEPS,
        add_noise: bool = False,
        noise_std: float = 0.05,
    ):
        """
        Initialize the SUMO ramp metering environment.
        
        Args:
            scenario: Traffic demand scenario ('stationary', 'flat_peak', 'sharp_peak')
            render_mode: Gymnasium render mode
            use_gui: Whether to use SUMO-GUI for visualization
            delta_time: Number of simulation steps per control action (default 120 steps)
            add_noise: Whether to add noise to demand during training
            noise_std: Standard deviation of demand noise
        """
        super().__init__()
        
        self.scenario = scenario
        self.render_mode = render_mode
        self.use_gui = use_gui
        self.delta_time = delta_time
        self.add_noise = add_noise
        self.noise_std = noise_std
        
        # SUMO configuration
        self.sumo_cfg = get_sumo_config(scenario)
        if use_gui:
            # Try to find sumo-gui in the same directory as SUMO_BINARY
            sumo_dir = os.path.dirname(SUMO_BINARY)
            gui_binary = os.path.join(sumo_dir, "sumo-gui")
            # Check for sumo-gui or sumo-gui.exe
            if os.path.exists(gui_binary + ".exe"):
                self.sumo_binary = gui_binary + ".exe"
            elif os.path.exists(gui_binary):
                self.sumo_binary = gui_binary
            else:
                # Fallback to system PATH
                self.sumo_binary = "sumo-gui"
        else:
            self.sumo_binary = SUMO_BINARY
        
        # Check if SUMO_HOME is set
        if 'SUMO_HOME' not in os.environ:
            # Try common paths
            for path in ['/usr/share/sumo', '/opt/sumo',
                         r'C:\Users\mprosperi\Desktop\sumo-1.26.0']:
                if os.path.exists(path):
                    os.environ['SUMO_HOME'] = path
                    break
        
        # Connection tracking
        self._conn_label = f"ramp_env_{id(self)}"
        self._sumo_running = False
        
        # State and action spaces
        # State: K occupancy values, normalized [0, 1]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(DETECTOR_PROFILES),),
            dtype=np.float32
        )
        
        # Action: Metering rate normalized to [0, 1]
        # Will be converted to actual rate in [MR_MIN, MR_MAX]
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_speeds = []
        self.episode_queues = []
        # Track speeds for all detector profiles
        self.episode_profile_speeds = {profile: [] for profile in DETECTOR_PROFILES}
        
        # Previous metering rate (for ALINEA and logging)
        self.prev_metering_rate = (MR_MIN + MR_MAX) / 2
        
    def _start_sumo(self):
        """Start SUMO simulation."""
        sumo_cmd = [
            self.sumo_binary,
            "-c", self.sumo_cfg,
            "--start",
            "--quit-on-end",
            "--no-warnings",
            "--no-step-log",
            "--step-length", str(SIM_STEP),  # Simulation step size
        ]
        
        traci.start(sumo_cmd, label=self._conn_label)
        self._sumo_running = True
        
    def _close_sumo(self):
        """Close SUMO connection."""
        if self._sumo_running:
            try:
                traci.switch(self._conn_label)
                traci.close()
            except traci.exceptions.FatalTraCIError:
                pass
            self._sumo_running = False
            
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment for a new episode.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options
            
        Returns:
            Initial observation and info dict
        """
        super().reset(seed=seed)
        
        # Close any existing simulation
        self._close_sumo()
        
        # Start new simulation
        self._start_sumo()
        traci.switch(self._conn_label)
        
        # Reset episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_speeds = []
        self.episode_queues = []
        # Track speeds for all detector profiles
        self.episode_profile_speeds = {profile: [] for profile in DETECTOR_PROFILES}
        self.prev_metering_rate = (MR_MIN + MR_MAX) / 2
        # Reset cached TLS link count for new SUMO instance
        if hasattr(self, '_num_tls_links'):
            del self._num_tls_links
        # Vehicle flow logging: list of (timestamp, edge, veh_type)
        self.episode_flows = []
        
        # Run simulation to first control period
        for _ in range(self.delta_time):
            traci.simulationStep()
            
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one control period step.
        
        Args:
            action: Normalized metering rate [0, 1]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        traci.switch(self._conn_label)
        
        # Convert action to metering rate
        metering_rate = self._action_to_metering_rate(action)
        
        # Apply metering rate (set traffic light timing)
        self._apply_metering_rate(metering_rate)
        self.prev_metering_rate = metering_rate
        
        # Advance simulation by one control period
        for _ in range(self.delta_time):
            traci.simulationStep()
            # Log vehicle flows for each step
            timestamp = traci.simulation.getTime()
            # Get all vehicles that entered the network in this step
            veh_ids = traci.simulation.getDepartedIDList()
            for veh_id in veh_ids:
                try:
                    edge = traci.vehicle.getRoadID(veh_id)
                    veh_type = traci.vehicle.getTypeID(veh_id)
                    self.episode_flows.append((timestamp, edge, veh_type))
                except Exception:
                    pass
            
        self.current_step += 1
        
        # Get new state
        observation = self._get_observation()
        
        # Calculate reward (eq. 3)
        reward = self._calculate_reward()
        self.episode_reward += reward
        
        # Check termination
        terminated = self.current_step >= NUM_CONTROL_PERIODS
        truncated = False
        
        # Get info
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current state observation (occupancies from detector profiles).
        Uses exact detector IDs from Additional.add.xml.
        
        Returns:
            Array of occupancy values, one per profile
        """
        occupancies = []
        
        for profile in DETECTOR_PROFILES:
            profile_occ = []
            detector_ids = get_detector_ids(profile)
            
            for det_id in detector_ids:
                try:
                    # Get last interval's occupancy
                    occ = traci.inductionloop.getLastIntervalOccupancy(det_id)
                    profile_occ.append(occ / 100.0)  # Normalize to [0, 1]
                except traci.exceptions.TraCIException:
                    profile_occ.append(0.0)
            
            # Average occupancy across lanes
            occupancies.append(np.mean(profile_occ) if profile_occ else 0.0)
        
        return np.array(occupancies, dtype=np.float32)
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward based on eq. 3: r(t+1) = -v_k'(t) - η*q(t)
        
        The paper uses negative reward to maximize speed and minimize queue.
        However, we negate this to get positive returns for better performance.
        
        Returns:
            Reward value
        """
        # Get speed at bottleneck (downstream profile)
        speed = self._get_bottleneck_speed()
        
        # Get queue size on ramp
        queue = self._get_ramp_queue()
        
        # Store for episode statistics
        self.episode_speeds.append(speed)
        self.episode_queues.append(queue)
        
        # Collect speeds for all detector profiles
        for profile in DETECTOR_PROFILES:
            profile_speed = self._get_profile_speed(profile)
            self.episode_profile_speeds[profile].append(profile_speed)
        
        # Reward function (modified to be positive for better performance)
        # Original: r = -speed - eta*queue (higher speed = less negative = better)
        # We use: r = speed - eta*queue (higher speed = more positive = better)
        reward = speed - ETA * queue
        
        return reward
    
    def _get_bottleneck_speed(self) -> float:
        """
        Get average speed at bottleneck profile (km/h).
        Uses detector IDs from Additional.add.xml.
        
        Returns:
            Average speed in km/h
        """
        return self._get_profile_speed(DOWNSTREAM_PROFILE)

    def _get_profile_speed(self, profile: str) -> float:
        """
        Get average speed for a detector profile (km/h).

        Args:
            profile: Detector profile name (e.g., 'p4', 'p5')

        Returns:
            Average speed in km/h
        """
        speeds = []
        detector_ids = get_detector_ids(profile)
        
        for det_id in detector_ids:
            try:
                speed_ms = traci.inductionloop.getLastIntervalMeanSpeed(det_id)
                if speed_ms >= 0:  # Valid speed
                    speeds.append(speed_ms * 3.6)  # Convert m/s to km/h
            except traci.exceptions.TraCIException:
                pass
        
        return np.mean(speeds) if speeds else 0.0
    
    def _get_ramp_queue(self) -> float:
        """
        Get queue length on ramp (meters).
        Uses queue_ramp laneAreaDetector from Additional.add.xml.
        
        Returns:
            Queue length in meters
        """
        try:
            # Use lane area detector for queue (ID from Additional.add.xml)
            queue = traci.lanearea.getJamLengthMeters(QUEUE_DETECTOR_ID)
            return queue
        except traci.exceptions.TraCIException:
            # Fallback: count waiting vehicles
            try:
                waiting = traci.edge.getLastStepHaltingNumber(EDGE_RAMP)
                return waiting * 5.0  # Approximate 5m per vehicle
            except:
                return 0.0
    
    def _action_to_metering_rate(self, action: np.ndarray) -> float:
        """
        Convert normalized action [0, 1] to metering rate [MR_MIN, MR_MAX].
        
        Args:
            action: Normalized action value
            
        Returns:
            Metering rate in veh/h
        """
        action_val = np.clip(action[0], 0.0, 1.0)
        return MR_MIN + action_val * (MR_MAX - MR_MIN)
    
    def _metering_rate_to_red_duration(self, metering_rate: float) -> float:
        """
        Convert metering rate to red phase duration (eq. 2 from paper).
        
        MR = 3600 / (green + red)
        red = 3600/MR - green
        
        Args:
            metering_rate: Metering rate in veh/h
            
        Returns:
            Red phase duration in seconds
        """
        if metering_rate <= 0:
            return MAX_RED
        
        cycle_time = 3600.0 / metering_rate
        red_duration = cycle_time - GREEN_PHASE
        return np.clip(red_duration, MIN_RED, MAX_RED)
    
    def _get_num_tls_links(self) -> int:
        """Get the number of controlled links at the ramp meter TLS."""
        if not hasattr(self, '_num_tls_links'):
            try:
                links = traci.trafficlight.getControlledLinks(TLS_ID)
                self._num_tls_links = len(links)
            except traci.exceptions.TraCIException:
                self._num_tls_links = 4  # fallback
        return self._num_tls_links

    def _apply_metering_rate(self, metering_rate: float):
        """
        Apply metering rate by setting traffic light phases.
        
        The ramp meter cycles between green and red-for-ramp.
        State strings are built dynamically based on the actual number of
        controlled links in the network (e.g. "GGGG" / "rGGG" for 4 links).
        
        Args:
            metering_rate: Metering rate in veh/h
        """
        n_links = self._get_num_tls_links()
        all_green = "G" * n_links       # e.g. "GGGG"
        ramp_red = "r" + "G" * (n_links - 1)  # e.g. "rGGG"
        
        if metering_rate >= MR_MAX:
            # No metering — set a permanent all-green phase.
            # IMPORTANT: Do NOT use setProgram("off") because that removes
            # the TLS control entirely, causing SUMO to fall back to the
            # junction's default priority rules where ramp and mainline get
            # equal right-of-way. By keeping the TLS active with all-green,
            # mainline retains priority over the ramp.
            phases = [traci.trafficlight.Phase(9999, all_green)]
            logic = traci.trafficlight.Logic(
                programID="no_control", type=0,
                currentPhaseIndex=0, phases=phases,
            )
            try:
                traci.trafficlight.setProgramLogic(TLS_ID, logic)
                traci.trafficlight.setProgram(TLS_ID, "no_control")
            except traci.exceptions.TraCIException:
                pass
            return
        
        if metering_rate <= MR_MIN:
            red_duration = MAX_RED
        else:
            # Normal metering: compute red duration from desired rate
            cycle_time = 3600.0 / metering_rate  # seconds per vehicle
            red_float = cycle_time - GREEN_PHASE
            red_duration = max(1, round(red_float))
            red_duration = min(red_duration, MAX_RED)
        
        phases = [
            traci.trafficlight.Phase(GREEN_PHASE, all_green),
            traci.trafficlight.Phase(red_duration, ramp_red),
        ]
        
        logic = traci.trafficlight.Logic(
            programID="metering",
            type=0,
            currentPhaseIndex=0,
            phases=phases
        )
        
        try:
            traci.trafficlight.setProgramLogic(TLS_ID, logic)
            traci.trafficlight.setProgram(TLS_ID, "metering")
        except traci.exceptions.TraCIException:
            pass
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional information about current state.
        
        Returns:
            Dictionary with info
        """
        return {
            'step': self.current_step,
            'episode_reward': self.episode_reward,
            'avg_speed': np.mean(self.episode_speeds) if self.episode_speeds else 0,
            'avg_queue': np.mean(self.episode_queues) if self.episode_queues else 0,
            'metering_rate': self.prev_metering_rate,
        }
    
    def render(self):
        """Render the environment (handled by SUMO-GUI if enabled)."""
        pass
    
    def close(self):
        """Clean up resources."""
        self._close_sumo()
    
    def get_episode_statistics(self) -> Dict[str, float]:
        """
        Get statistics for the completed episode.
        
        Returns:
            Dictionary with episode statistics matching Table I in paper
        """
        stats = {
            'total_return': self.episode_reward,
            'avg_speed_kmh': np.mean(self.episode_speeds) if self.episode_speeds else 0,
            'avg_queue_m': np.mean(self.episode_queues) if self.episode_queues else 0,
            'max_queue_m': max(self.episode_queues) if self.episode_queues else 0,
            'num_steps': self.current_step,
        }
        # Add per-profile speed statistics
        for profile in DETECTOR_PROFILES:
            speeds = self.episode_profile_speeds.get(profile, [])
            stats[f'avg_speed_{profile}_kmh'] = np.mean(speeds) if speeds else 0.0
        # Add vehicle flows for episode
        stats['vehicle_flows'] = self.episode_flows
        return stats


# Register the environment
gym.register(
    id='SumoRampMetering-v0',
    entry_point='sumo_env:SumoRampMeteringEnv',
)
