"""
Base controller class for ramp metering algorithms.
All controllers inherit from this abstract base class.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional


class BaseController(ABC):
    """
    Abstract base class for ramp metering controllers.
    
    Each controller takes the current state (occupancies) and returns
    a metering rate action.
    """
    
    def __init__(self, name: str = "base"):
        """
        Initialize the controller.
        
        Args:
            name: Controller name for logging/identification
        """
        self.name = name
        self._step_count = 0
        
    @abstractmethod
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute the metering action given current state.
        
        Args:
            observation: State observation (occupancies from detector profiles)
        
        Returns:
            Action array (normalized metering rate in [0, 1])
        """
        pass
    
    def reset(self):
        """Reset controller state for new episode."""
        self._step_count = 0
    
    def step(self):
        """Called after each control step."""
        self._step_count += 1
    
    def update(self, observation: np.ndarray, action: np.ndarray, 
               reward: float, next_observation: np.ndarray, done: bool):
        """
        Update controller with transition data (for learning controllers).
        
        Args:
            observation: Current state
            action: Action taken
            reward: Reward received
            next_observation: Next state
            done: Whether episode ended
        """
        pass  # Default: no learning
    
    def save(self, path: str):
        """Save controller state/model."""
        pass
    
    def load(self, path: str):
        """Load controller state/model."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
