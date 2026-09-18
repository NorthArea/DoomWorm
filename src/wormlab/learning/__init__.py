"""Fitness, evolution, plasticity. Topology is fixed; only weights are trainable."""

from wormlab.learning.bakeoff import TrainConfig, fitness_of, train_candidate
from wormlab.learning.benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    leaderboard,
    run_benchmark,
    save_result,
    write_leaderboard,
)
from wormlab.learning.evolution import (
    EvolutionConfig,
    EvolutionResult,
    GenerationStats,
    evolve,
    seeded_population,
)
from wormlab.learning.fitness import Evaluation, Scenario, evaluate
from wormlab.learning.reward import RewardConfig, RewardTracker

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "Evaluation",
    "EvolutionConfig",
    "EvolutionResult",
    "GenerationStats",
    "RewardConfig",
    "RewardTracker",
    "Scenario",
    "TrainConfig",
    "evaluate",
    "evolve",
    "fitness_of",
    "leaderboard",
    "run_benchmark",
    "save_result",
    "seeded_population",
    "train_candidate",
    "write_leaderboard",
]
