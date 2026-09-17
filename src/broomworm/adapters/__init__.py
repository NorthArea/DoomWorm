"""Sensory adapter (observations -> stimulation) and motor adapter (activity -> actions)."""

from broomworm.adapters.motor import GroupMotorAdapter, MotorAdapter
from broomworm.adapters.sensory import SensoryAdapter

__all__ = ["GroupMotorAdapter", "MotorAdapter", "SensoryAdapter"]
