"""Fitness, evolution, plasticity. Topology is fixed; only weights are trainable."""

from doomworm.learning.evolution import EvolutionConfig, EvolutionResult, GenerationStats, evolve
from doomworm.learning.fitness import Evaluation, Scenario, evaluate
from doomworm.learning.reward import RewardConfig, RewardTracker

__all__ = [
    "Evaluation",
    "EvolutionConfig",
    "EvolutionResult",
    "GenerationStats",
    "RewardConfig",
    "RewardTracker",
    "Scenario",
    "evaluate",
    "evolve",
]
