"""Evolutionary weight search (Plan §10).

    create population -> evaluate -> keep elites -> copy + mutate -> repeat

Topology is never touched: a genome is the weight vector of a fixed network.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

Fitness = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class EvolutionConfig:
    """Hyper-parameters of the search."""

    population: int = 60
    generations: int = 30
    elite_fraction: float = 0.2
    mutation_sigma: float = 0.3
    weight_range: tuple[float, float] = (-2.0, 2.0)

    @property
    def n_elite(self) -> int:
        """Number of genomes copied unchanged into the next generation."""
        return max(1, int(self.population * self.elite_fraction))


@dataclass(frozen=True)
class GenerationStats:
    """Fitness summary of one generation."""

    generation: int
    best: float
    mean: float
    worst: float


@dataclass
class EvolutionResult:
    """Outcome of :func:`evolve`."""

    best_weights: np.ndarray
    best_fitness: float
    history: list[GenerationStats] = field(default_factory=list)


def random_population(n_weights: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform genomes in [-1, 1]."""
    return rng.uniform(-1.0, 1.0, size=(size, n_weights))


def seeded_population(
    initial: np.ndarray, size: int, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """``initial`` unchanged plus ``size - 1`` Gaussian perturbations of it."""
    noise = rng.normal(0.0, sigma, size=(size - 1, initial.size))
    return np.vstack([initial[None, :], initial[None, :] + noise])


def evolve(
    fitness: Fitness,
    n_weights: int,
    config: EvolutionConfig | None = None,
    seed: int = 0,
    on_generation: Callable[[GenerationStats], None] | None = None,
    initial: np.ndarray | None = None,
    initial_sigma: float | None = None,
) -> EvolutionResult:
    """Run truncation selection with Gaussian mutation; deterministic for a seed.

    With ``initial`` the first generation is that genome plus perturbations of
    scale ``initial_sigma`` (default: the mutation sigma) instead of uniform
    random genomes: this is how a connectome's weights are trained in place.
    """
    cfg = config or EvolutionConfig()
    rng = np.random.default_rng(seed)
    if initial is None:
        population = random_population(n_weights, cfg.population, rng)
    else:
        if initial.size != n_weights:
            raise ValueError(f"initial genome has {initial.size} weights, expected {n_weights}")
        sigma0 = cfg.mutation_sigma if initial_sigma is None else initial_sigma
        population = seeded_population(
            np.asarray(initial, dtype=float), cfg.population, sigma0, rng
        )
    low, high = cfg.weight_range

    best_weights = population[0].copy()
    best_fitness = -np.inf
    history: list[GenerationStats] = []

    for generation in range(cfg.generations):
        scores = np.array([fitness(genome) for genome in population])
        order = np.argsort(-scores)
        population = population[order]
        scores = scores[order]

        if scores[0] > best_fitness:
            best_fitness = float(scores[0])
            best_weights = population[0].copy()

        stats = GenerationStats(
            generation=generation,
            best=float(scores[0]),
            mean=float(scores.mean()),
            worst=float(scores[-1]),
        )
        history.append(stats)
        if on_generation is not None:
            on_generation(stats)

        elites = population[: cfg.n_elite]
        n_children = cfg.population - cfg.n_elite
        parents = elites[rng.integers(0, len(elites), size=n_children)]
        children = parents + rng.normal(0.0, cfg.mutation_sigma, size=parents.shape)
        population = np.vstack([elites, np.clip(children, low, high)])

    return EvolutionResult(best_weights=best_weights, best_fitness=best_fitness, history=history)
