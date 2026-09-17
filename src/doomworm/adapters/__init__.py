"""Sensory adapter (observations -> stimulation) and motor adapter (activity -> actions)."""

from doomworm.adapters.motor import FireAdapter, GroupMotorAdapter, MotorAdapter
from doomworm.adapters.sensory import SensoryAdapter

__all__ = ["FireAdapter", "GroupMotorAdapter", "MotorAdapter", "SensoryAdapter"]
