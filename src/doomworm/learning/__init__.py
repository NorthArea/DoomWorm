"""Fitness, evolution, plasticity. Topology is fixed; only weights are trainable."""

from doomworm.learning.bakeoff import TrainConfig, fitness_of, train_candidate
from doomworm.learning.benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    leaderboard,
    run_benchmark,
    save_result,
    write_leaderboard,
)
from doomworm.learning.evolution import (
    EvolutionConfig,
    EvolutionResult,
    GenerationStats,
    evolve,
    seeded_population,
)
from doomworm.learning.fitness import Evaluation, Scenario, evaluate
from doomworm.learning.reward import RewardConfig, RewardTracker
from doomworm.learning.robustness import CAR_GRID, SweepRow, sweep, sweep_table, worst_cells

__all__ = [
    "CAR_GRID",
    "BenchmarkConfig",
    "BenchmarkResult",
    "Evaluation",
    "EvolutionConfig",
    "EvolutionResult",
    "GenerationStats",
    "RewardConfig",
    "RewardTracker",
    "Scenario",
    "SweepRow",
    "TrainConfig",
    "evaluate",
    "evolve",
    "fitness_of",
    "leaderboard",
    "run_benchmark",
    "save_result",
    "seeded_population",
    "sweep",
    "sweep_table",
    "train_candidate",
    "worst_cells",
    "write_leaderboard",
]
