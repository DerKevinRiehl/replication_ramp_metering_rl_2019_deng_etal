"""
ALINEA ramp metering controller.
Implementation of the classic ALINEA algorithm (Papageorgiou et al., 1991).

From Section III.B.2: "The state-of-the-art ramp metering algorithm ALINEA 
is used as benchmark. We set the critical occupancy to 0.18 and the metering 
gain KR to 0.35 (as recommended in [23])."

ALINEA Formula:
    MR(t) = MR(t-1) + K_R * (O_cr - O_down(t))

Where:
    - MR(t): Metering rate at time t
    - K_R: Metering gain (regulator parameter)
    - O_cr: Critical (desired) occupancy
    - O_down(t): Measured downstream occupancy (mainline lanes only at p4)
"""

import numpy as np
import traci
from .base_controller import BaseController
from config import (
    ALINEA_O_CRIT, ALINEA_KR,
    MR_MIN, MR_MAX, DETECTOR_PROFILES, DOWNSTREAM_PROFILE,
    SCENARIOS
)

# Mainline-only detectors at p4 (excluding acceleration lane det_main_p4_l0).
# Classical ALINEA reads mainline occupancy at the merge area.
ALINEA_P4_MAINLINE_DETS = ['det_main_p4_l1', 'det_main_p4_l2', 'det_main_p4_l3']


class AlineaController(BaseController):
    """
    ALINEA (Asservissement Linéaire d'Entrée Autoroutière) controller.
    
    Local feedback control that adjusts metering rate based on the
    difference between measured downstream occupancy and a target
    critical occupancy.

    Classical ALINEA reads mainline-only occupancy at the merge area (p4),
    excluding the acceleration lane detector which biases the reading.
    
    The update rule in normalised action space [0, 1]:
        a(t) = a(t-1) + K_R * (O_cr - O_down(t))
    where K_R = 0.35, O_cr = 0.18.
    The actual metering rate is then:
        MR = MR_MIN + a * (MR_MAX - MR_MIN)
    """
    
    def __init__(
        self,
        scenario: str = 'stationary',
        o_crit: float = None,
        k_r: float = None,
        initial_action: float = 0.5
    ):
        """
        Initialize ALINEA controller.
        
        Args:
            scenario: Traffic scenario — used to look up calibrated O_crit / KR
                      from config.ALINEA_O_CRIT / ALINEA_KR dicts.
            o_crit: Override critical occupancy [0, 1]. If None, uses config value.
            k_r: Override metering gain. If None, uses config value.
            initial_action: Initial normalised action [0, 1]. Default: 0.5 (mid-range).
        """
        super().__init__(name="alinea")

        # Resolve O_crit from per-scenario dict or explicit override
        if o_crit is not None:
            self.o_crit = o_crit
        else:
            val = ALINEA_O_CRIT.get(scenario) if isinstance(ALINEA_O_CRIT, dict) else ALINEA_O_CRIT
            if val is None:
                raise ValueError(
                    f"ALINEA_O_CRIT for scenario '{scenario}' is not set in config.py. "
                    "Run sweep_ocrit.py and fill in the value."
                )
            self.o_crit = val

        # Resolve KR from per-scenario dict or explicit override
        if k_r is not None:
            self.k_r = k_r
        else:
            val = ALINEA_KR.get(scenario) if isinstance(ALINEA_KR, dict) else ALINEA_KR
            self.k_r = val if val is not None else 0.35
        self.initial_action = initial_action
        
        # Current normalised action [0, 1]
        self.current_action = initial_action
        
    def reset(self):
        """Reset controller for new episode."""
        super().reset()
        self.current_action = self.initial_action
    
    def _read_mainline_p4_occupancy(self) -> float:
        """
        Read average occupancy from mainline-only p4 detectors via TraCI.
        Excludes the acceleration lane detector which biases the reading.
        
        Returns:
            Average mainline occupancy at p4 in [0, 1]
        """
        occs = []
        for det_id in ALINEA_P4_MAINLINE_DETS:
            try:
                occ = traci.inductionloop.getLastIntervalOccupancy(det_id)
                occs.append(occ / 100.0)  # Convert to [0, 1]
            except traci.exceptions.TraCIException:
                pass
        return np.mean(occs) if occs else 0.0

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute ALINEA metering action.
        
        Args:
            observation: State observation (occupancies from detector profiles)
        
        Returns:
            Normalized metering rate action [0, 1]
        """
        # Read mainline-only p4 occupancy directly from TraCI
        # (bypasses agent observation which averages all lanes including acc lane)
        o_down = self._read_mainline_p4_occupancy()
        
        # ALINEA update rule in normalised space (Section III.B.2):
        #   a(t) = a(t-1) + K_R * (O_cr - O_down(t))
        # K_R = 0.35, O_cr = 0.18 (as stated in the paper)
        delta = self.k_r * (self.o_crit - o_down)
        
        # Update normalised action
        self.current_action = self.current_action + delta
        
        # Clamp to [0, 1]
        self.current_action = float(np.clip(self.current_action, 0.0, 1.0))
        
        return np.array([self.current_action], dtype=np.float32)
