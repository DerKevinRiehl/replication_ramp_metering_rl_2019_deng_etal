"""
Controllers package for ramp metering algorithms.
"""

from typing import TYPE_CHECKING

from .base_controller import BaseController
from .no_control import NoControlController
from .fixed_time import FixedTimeController
from .alinea import AlineaController

if TYPE_CHECKING:
    from .ppo_controller import PPOController

__all__ = [
    'BaseController',
    'NoControlController',
    'FixedTimeController',
    'AlineaController',
    'PPOController',
]


def __getattr__(name: str):
    if name == 'PPOController':
        from .ppo_controller import PPOController

        return PPOController
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
