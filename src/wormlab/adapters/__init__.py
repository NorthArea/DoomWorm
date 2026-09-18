"""Sensory adapter (observations -> stimulation) and motor adapter (activity -> actions)."""

from wormlab.adapters.motor import FireAdapter, GroupMotorAdapter, MotorAdapter
from wormlab.adapters.sensory import SensoryAdapter

__all__ = ["FireAdapter", "GroupMotorAdapter", "MotorAdapter", "SensoryAdapter"]
