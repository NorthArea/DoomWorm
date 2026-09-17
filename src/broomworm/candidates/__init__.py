"""Candidate brains behind one interface (Plan §3.3, §20.4).

The connectome worm and its topology controls, the small recurrent net, the
Neural Circuit Policy, the Roomba-style classical controller, the hybrid (a frozen
reflex under a trainable policy), and the PPO
descriptor (loaded through :func:`load_candidate`, optional ``rl`` group).
"""

from broomworm.candidates.base import Brain, ScriptedBrain, Trainable, Wheels
from broomworm.candidates.hybrid import HybridBrain
from broomworm.candidates.registry import CANDIDATES, CandidateSpec, build_candidate, load_candidate
from broomworm.candidates.rnn import RNNBrain
from broomworm.candidates.roomba import RoombaBrain
from broomworm.candidates.worm import WormBrain

__all__ = [
    "CANDIDATES",
    "Brain",
    "CandidateSpec",
    "HybridBrain",
    "RNNBrain",
    "RoombaBrain",
    "ScriptedBrain",
    "Trainable",
    "Wheels",
    "WormBrain",
    "build_candidate",
    "load_candidate",
]
