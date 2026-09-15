"""Candidate brains behind one interface (Plan §3.3)."""

from doomworm.brains.base import Brain, ScriptedBrain, Trainable, Wheels
from doomworm.brains.candidates import CANDIDATES, CandidateSpec, build_candidate, load_candidate
from doomworm.brains.needs import NeedsArbiter
from doomworm.brains.planner_layer import PlannerLayer, sector_channels
from doomworm.brains.rnn import RNNBrain
from doomworm.brains.roomba import RoombaBrain
from doomworm.brains.simple import GradientFollower
from doomworm.brains.worm import WormBrain

__all__ = [
    "CANDIDATES",
    "Brain",
    "CandidateSpec",
    "GradientFollower",
    "NeedsArbiter",
    "PlannerLayer",
    "RNNBrain",
    "RoombaBrain",
    "ScriptedBrain",
    "Trainable",
    "Wheels",
    "WormBrain",
    "build_candidate",
    "load_candidate",
    "sector_channels",
]
