"""
No-control baseline: ramp meter always green (no metering).
Used as baseline comparison in Table I of Deng et al. (2019).
"""

import numpy as np
from .base_controller import BaseController


class NoControlController(BaseController):
    """
    No-control strategy: Maximum metering rate (always green).
    
    This baseline allows all vehicles to enter freely without any
    metering restriction. It typically results in:
    - Minimum queue on ramp (vehicles enter immediately)
    - Lower mainline speeds (potential congestion at merge)
    """
    
    def __init__(self):
        super().__init__(name="no_control")
    
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Return maximum metering rate (no restriction).
        
        Args:
            observation: State observation (ignored)
        
        Returns:
            Action = 1.0 (maximum metering rate)
        """
        # Return maximum normalized action (= maximum metering rate)
        return np.array([1.0], dtype=np.float32)
