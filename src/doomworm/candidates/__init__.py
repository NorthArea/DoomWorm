"""Candidate brains behind one interface (Plan §3.3, §20.4).

The connectome worm and its topology controls, the small recurrent net, the
Neural Circuit Policy, the Roomba-style classical controller, the hand-written
Doom player and the PPO
descriptor (loaded through :func:`load_candidate`, optional ``rl`` group).
"""

from doomworm.candidates.base import Brain, ScriptedBrain, Trainable, Wheels
from doomworm.candidates.doomguy import DoomguyBrain
from doomworm.candidates.registry import CANDIDATES, CandidateSpec, build_candidate, load_candidate
from doomworm.candidates.rnn import RNNBrain
from doomworm.candidates.roomba import RoombaBrain
from doomworm.candidates.worm import WormBrain

__all__ = [
    "CANDIDATES",
    "Brain",
    "CandidateSpec",
    "DoomguyBrain",
    "RNNBrain",
    "RoombaBrain",
    "ScriptedBrain",
    "Trainable",
    "Wheels",
    "WormBrain",
    "build_candidate",
    "load_candidate",
]
