"""Stage 18: the optimiser has to be stronger than the problem, or a negative result is ours."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pytest

from doomworm.learning.cmaes import CMAConfig, cma_es


def sphere(x: np.ndarray) -> float:
    """Maximised at the origin."""
    return -float(np.sum(x**2))


def ellipsoid(x: np.ndarray) -> float:
    """Badly scaled but separable: coordinate i matters 10^6 times more than coordinate 0.

    This is the property the separable form exists for -- a per-coordinate step.
    (Rosenbrock would be the wrong test: it is non-separable by construction, and
    sep-CMA-ES is not claimed to solve it.)
    """
    n = x.size
    powers = 10.0 ** (6.0 * np.arange(n) / max(1, n - 1))
    return -float(np.sum(powers * x**2))


def fixed_sigma_search(
    fitness: Callable[[np.ndarray], float],
    initial: np.ndarray,
    generations: int,
    population: int,
    sigma: float,
    seed: int,
) -> float:
    """What the project used before: mutate the best by a fixed sigma, keep the best."""
    rng = np.random.default_rng(seed)
    best = np.asarray(initial, dtype=float)
    best_score = fitness(best)
    for _ in range(generations):
        for candidate in best + rng.normal(0.0, sigma, size=(population, best.size)):
            score = fitness(candidate)
            if score > best_score:
                best, best_score = candidate, score
    return best_score


def test_it_actually_finds_the_optimum() -> None:
    n = 20
    start = np.full(n, 0.5)
    result = cma_es(sphere, start, CMAConfig(generations=120, sigma=0.3, seed=1))
    assert result.best_fitness > -1e-4, "the sphere is solved to numerical noise"
    assert np.allclose(result.best_weights, 0.0, atol=5e-3)
    assert len(result.history) == 120
    assert result.history[-1].sigma < result.history[0].sigma, "the step contracts on success"


def test_it_beats_the_fixed_sigma_search_it_replaces() -> None:
    """The point of the change: the same budget, a much better answer."""
    n, generations, population, seed = 20, 60, 12, 3
    start = np.full(n, 0.5)
    theirs = fixed_sigma_search(sphere, start, generations, population, 0.02, seed)
    cfg = CMAConfig(generations=generations, population=population, sigma=0.3, seed=seed)
    ours = cma_es(sphere, start, cfg).best_fitness
    assert ours > theirs, f"cma {ours:.4f} should beat fixed sigma {theirs:.4f}"
    assert ours > -0.05, "and should get within a whisker of the optimum"


def test_it_handles_a_badly_scaled_problem() -> None:
    """Where one weight matters a million times more than another, and a fixed step cannot."""
    n, generations, population, seed = 10, 300, 12, 2
    start = np.full(n, 0.5)
    ours = cma_es(
        ellipsoid,
        start,
        CMAConfig(generations=generations, population=population, sigma=0.3, seed=seed),
    ).best_fitness
    theirs = fixed_sigma_search(ellipsoid, start, generations, population, 0.02, seed)
    assert ours > theirs, f"cma {ours:.1f} should beat fixed sigma {theirs:.1f}"
    assert ours > -1.0, f"and should get close to the optimum, got {ours:.3f}"


def test_it_respects_the_weight_range_and_is_deterministic() -> None:
    cfg = CMAConfig(generations=10, sigma=0.5, weight_range=(-0.2, 0.2), seed=7)
    a = cma_es(sphere, np.full(8, 0.1), cfg)
    b = cma_es(sphere, np.full(8, 0.1), cfg)
    assert a.best_fitness == b.best_fitness, "same seed, same answer"
    assert np.all(np.abs(a.best_weights) <= 0.2 + 1e-12)


def test_the_population_grows_with_the_problem() -> None:
    assert CMAConfig().lambda_for(10) == 10
    assert CMAConfig().lambda_for(1819) > CMAConfig().lambda_for(506)
    assert CMAConfig(population=40).lambda_for(5905) == 40


def test_it_reports_every_generation() -> None:
    seen: list[int] = []
    cma_es(
        sphere,
        np.full(5, 0.3),
        CMAConfig(generations=6, seed=0),
        on_generation=lambda g, _w, _f: seen.append(g.generation),
    )
    assert seen == list(range(6))


def test_a_pool_may_evaluate_the_population() -> None:
    calls: list[int] = []

    def batched(fn: Callable[[np.ndarray], float], xs: Sequence[np.ndarray]) -> list[float]:
        calls.append(len(xs))
        return [fn(x) for x in xs]

    result = cma_es(sphere, np.full(4, 0.2), CMAConfig(generations=3, seed=0), map_fn=batched)
    assert len(calls) == 3
    assert result.best_fitness == pytest.approx(result.best_fitness)
