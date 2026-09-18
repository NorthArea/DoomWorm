"""Candidate brains behind one interface (Plan §3.3, §20.4).

The connectome worm and its topology controls, the small recurrent net, the
Neural Circuit Policy, the gradient follower,
descriptor (loaded through :func:`load_candidate`, optional ``rl`` group).
"""

from wormlab.candidates.base import Brain, ScriptedBrain, Trainable, Wheels
from wormlab.candidates.follower import GradientFollower
from wormlab.candidates.hybrid import HybridBrain
from wormlab.candidates.registry import CANDIDATES, CandidateSpec, build_candidate, load_candidate
from wormlab.candidates.rnn import RNNBrain
from wormlab.candidates.worm import WormBrain

__all__ = [
    "CANDIDATES",
    "Brain",
    "CandidateSpec",
    "GradientFollower",
    "HybridBrain",
    "RNNBrain",
    "ScriptedBrain",
    "Trainable",
    "Wheels",
    "WormBrain",
    "build_candidate",
    "load_candidate",
]
