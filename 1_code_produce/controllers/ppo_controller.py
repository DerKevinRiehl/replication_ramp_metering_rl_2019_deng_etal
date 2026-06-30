"""
PPO-based ramp metering controller.
Wrapper for Stable-Baselines3 PPO agent as described in Section II.C.

The algorithm uses Proximal Policy Optimization (PPO) with:
- Actor-Critic architecture (Figure 2)
- Continuous action space (metering rate)
- State: occupancy values from K detector profiles
- Reward: speed - eta * queue (eq. 3)
"""

import numpy as np
from pathlib import Path
from typing import Optional
from .base_controller import BaseController

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    PPO = None

from config import (
    PPO_POLICY, PPO_NET_ARCH, PPO_LEARNING_RATE, PPO_LOG_STD_INIT,
    PPO_N_STEPS, PPO_BATCH_SIZE, PPO_N_EPOCHS,
    PPO_GAMMA, PPO_GAE_LAMBDA, PPO_CLIP_RANGE,
    PPO_ENT_COEF, PPO_VF_COEF, PPO_MAX_GRAD_NORM,
    get_model_path, MODELS_DIR
)


class PPOController(BaseController):
    """
    PPO-based ramp metering controller using Stable-Baselines3.
    
    This controller wraps a trained PPO model for inference.
    Training is done separately via train_ppo.py.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        scenario: str = "stationary",
        deterministic: bool = True
    ):
        """
        Initialize PPO controller.
        
        Args:
            model_path: Path to trained model. If None, uses default path.
            scenario: Scenario name (for default model path)
            deterministic: Whether to use deterministic actions
        """
        super().__init__(name="ppo")
        
        if not SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 is required for PPO controller. "
                "Install with: pip install stable-baselines3"
            )
        
        self.scenario = scenario
        self.deterministic = deterministic
        self.model = None
        
        # Load model if path provided or default exists
        if model_path is not None:
            self.load(model_path)
        else:
            default_path = get_model_path(scenario)
            if default_path.exists():
                self.load(str(default_path))
    
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Get action from PPO policy.
        
        Args:
            observation: State observation (occupancies)
        
        Returns:
            Normalized metering rate action [0, 1]
        """
        if self.model is None:
            raise RuntimeError(
                "No model loaded. Train a model first using train_ppo.py "
                "or load an existing model."
            )
        
        # SB3 expects observation shape (n_envs, obs_dim)
        obs = observation.reshape(1, -1)
        
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        
        # Ensure action is in valid range
        action = np.clip(action, 0.0, 1.0)
        
        return action.flatten()
    
    def load(self, path: str):
        """
        Load trained model from file.
        
        Args:
            path: Path to model file (.zip)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        
        self.model = PPO.load(str(path))
        print(f"Loaded PPO model from {path}")
    
    def save(self, path: str):
        """
        Save model to file.
        
        Args:
            path: Path to save model (.zip)
        """
        if self.model is None:
            raise RuntimeError("No model to save")
        
        self.model.save(path)
        print(f"Saved PPO model to {path}")
    
    @staticmethod
    def create_model(env, tensorboard_log: Optional[str] = None):
        """
        Create a new PPO model with paper-specified hyperparameters.
        
        Args:
            env: Gymnasium environment
            tensorboard_log: Directory for TensorBoard logs
        
        Returns:
            PPO model instance
        """
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 required")
        
        policy_kwargs = {
            "net_arch": PPO_NET_ARCH,       # dict(pi=[64,64], vf=[64,64])
            "log_std_init": PPO_LOG_STD_INIT,  # std ≈ 0.37 instead of 1.0
        }
        
        model = PPO(
            policy=PPO_POLICY,
            env=env,
            learning_rate=PPO_LEARNING_RATE,
            n_steps=PPO_N_STEPS,
            batch_size=PPO_BATCH_SIZE,
            n_epochs=PPO_N_EPOCHS,
            gamma=PPO_GAMMA,
            gae_lambda=PPO_GAE_LAMBDA,
            clip_range=PPO_CLIP_RANGE,
            ent_coef=PPO_ENT_COEF,
            vf_coef=PPO_VF_COEF,
            max_grad_norm=PPO_MAX_GRAD_NORM,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=1,
        )
        
        return model
    
    def is_loaded(self) -> bool:
        """Check if a model is loaded."""
        return self.model is not None
