"""Candidate brains behind one interface (Plan §3.3)."""

from doomworm.brains.base import Brain, ScriptedBrain, Wheels
from doomworm.brains.needs import NeedsArbiter
from doomworm.brains.planner_layer import PlannerLayer, sector_channels
from doomworm.brains.roomba import RoombaBrain
from doomworm.brains.simple import GradientFollower
from doomworm.brains.worm import WormBrain

__all__ = [
    "Brain",
    "GradientFollower",
    "NeedsArbiter",
    "PlannerLayer",
    "RoombaBrain",
    "ScriptedBrain",
    "Wheels",
    "WormBrain",
    "sector_channels",
]
