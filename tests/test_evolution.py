"""Evolutionary weight search (Plan §10)."""

import numpy as np
import pytest

from broomworm.learning import EvolutionConfig, evolve


def test_evolve_finds_target_vector() -> None:
    target = np.array([0.5, -1.0, 1.5])

    def fitness(w: np.ndarray) -> float:
        return -float(np.sum((w - target) ** 2))

    cfg = EvolutionConfig(population=40, generations=40, elite_fraction=0.25, mutation_sigma=0.2)
    result = evolve(fitness, n_weights=3, config=cfg, seed=1)
    assert result.best_fitness > -0.05
    assert result.best_weights == pytest.approx(target, abs=0.15)
    assert len(result.history) == cfg.generations
    assert result.history[-1].best >= result.history[0].best


def test_evolve_is_deterministic() -> None:
    def fitness(w: np.ndarray) -> float:
        return -float(np.sum(w**2))

    cfg = EvolutionConfig(population=10, generations=5)
    a = evolve(fitness, n_weights=4, config=cfg, seed=3)
    b = evolve(fitness, n_weights=4, config=cfg, seed=3)
    assert a.best_weights == pytest.approx(b.best_weights)
    assert [g.best for g in a.history] == [g.best for g in b.history]


def test_elites_survive_unchanged() -> None:
    calls: list[np.ndarray] = []

    def fitness(w: np.ndarray) -> float:
        calls.append(w.copy())
        return float(w[0])

    cfg = EvolutionConfig(population=8, generations=3, elite_fraction=0.25, mutation_sigma=0.1)
    result = evolve(fitness, n_weights=1, config=cfg, seed=0)
    assert result.history[1].best >= result.history[0].best
    assert result.history[2].best >= result.history[1].best
    assert len(calls) == cfg.population * cfg.generations
