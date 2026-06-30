"""
Fixed-time ramp metering controller.
Uses a pre-determined constant metering rate throughout the simulation.

From Section III.B.2: "The proper metering rates of fixed-time control 
are determined by enumeration experiment in different scenarios."
"""

import numpy as np
from .base_controller import BaseController
from config import FIXED_TIME_RATES, MR_MIN, MR_MAX


class FixedTimeController(BaseController):
    """
    Fixed-time ramp metering strategy.
    
    Uses a constant metering rate determined offline through calibration.
    The rate can be scenario-specific to optimize performance.
    """
    
    def __init__(self, scenario: str = "stationary", metering_rate: float = None):
        """
        Initialize fixed-time controller.
        
        Args:
            scenario: Traffic scenario for rate selection
            metering_rate: Override metering rate (veh/h). If None, uses default.
        """
        super().__init__(name="fixed_time")
        
        self.scenario = scenario
        
        # Set metering rate
        if metering_rate is not None:
            self.metering_rate = metering_rate
        else:
            self.metering_rate = FIXED_TIME_RATES.get(scenario, 500)
        
        # Normalize to [0, 1]
        self.normalized_rate = (self.metering_rate - MR_MIN) / (MR_MAX - MR_MIN)
        self.normalized_rate = np.clip(self.normalized_rate, 0.0, 1.0)
        
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Return constant metering rate.
        
        Args:
            observation: State observation (ignored)
        
        Returns:
            Fixed normalized metering rate
        """
        return np.array([self.normalized_rate], dtype=np.float32)
    
    def set_metering_rate(self, rate: float):
        """
        Update the fixed metering rate.
        
        Args:
            rate: New metering rate in veh/h
        """
        self.metering_rate = rate
        self.normalized_rate = (rate - MR_MIN) / (MR_MAX - MR_MIN)
        self.normalized_rate = np.clip(self.normalized_rate, 0.0, 1.0)
