"""Sensory adapter (observations -> stimulation) and motor adapter (activity -> actions)."""

from doomworm.adapters.motor import MotorAdapter
from doomworm.adapters.sensory import SensoryAdapter

__all__ = ["MotorAdapter", "SensoryAdapter"]
