"""sep-CMA-ES: a search strong enough that a negative result means something.

The stage-4 evolution moves every weight by a fixed sigma and keeps the best.
In 5905 dimensions with a thousand evaluations that is not a search, and any
"the connectome cannot do this" conclusion drawn from it would really say "our
mutation could not find it".

This is the separable form of CMA-ES (Ros & Hansen 2008): the covariance is
kept diagonal, so memory and time are linear in the number of weights instead
of quadratic -- 5905 numbers instead of 35 million -- while the step size and
the per-coordinate scale still adapt. It keeps the parts that matter here:

    * an adaptive step size, so the search opens up when progress is steady and
      contracts when it is not;
    * per-coordinate scaling, so a weight that matters gets a wider stride than
      one that does not;
    * rank-based weighting, so the update ignores how *much* better a good
      sample was -- only that it was better, which suits a noisy benchmark.

Maximisation, to match `fitness_of`: the caller returns reward, higher better.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

FitnessFn = Callable[[np.ndarray], float]
MapFn = Callable[[Callable[[np.ndarray], float], Sequence[np.ndarray]], Sequence[float]]


@dataclass(frozen=True)
class CMAConfig:
    """Search settings. The defaults are the published ones for the separable form."""

    generations: int = 25
    population: int = 0  # 0 = the standard 4 + floor(3 ln n)
    sigma: float = 0.2  # initial step, in weight units
    weight_range: tuple[float, float] = (-1.0, 1.0)
    seed: int = 0

    def lambda_for(self, n: int) -> int:
        """Population size for a problem of ``n`` dimensions."""
        return self.population or 4 + int(3 * math.log(n))


@dataclass
class CMAGeneration:
    """What one generation did, for the log and the csv."""

    generation: int
    best: float
    mean: float
    worst: float
    sigma: float


@dataclass
class CMAResult:
    """The best genome found and the trace that got there."""

    best_weights: np.ndarray
    best_fitness: float
    history: list[CMAGeneration] = field(default_factory=list)


def _default_map(fn: Callable[[np.ndarray], float], xs: Sequence[np.ndarray]) -> list[float]:
    return [fn(x) for x in xs]


def cma_es(
    fitness: FitnessFn,
    initial: Sequence[float] | np.ndarray,
    config: CMAConfig | None = None,
    on_generation: Callable[[CMAGeneration, np.ndarray, float], None] | None = None,
    map_fn: MapFn | None = None,
) -> CMAResult:
    """Maximise ``fitness`` from ``initial`` with separable CMA-ES.

    ``fitness`` is called across ``map_fn`` when one is given, so it must be
    picklable for a process pool: no closures.
    """
    cfg = config or CMAConfig()
    mean = np.asarray(initial, dtype=float).copy()
    n = mean.size
    lam = cfg.lambda_for(n)
    mu = lam // 2

    # rank weights: the best sample counts most, and the worst half not at all
    raw = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = raw / raw.sum()
    mu_eff = 1.0 / float(np.sum(weights**2))

    # the separable constants: rank-one and rank-mu learning rates scaled by (n+2)/3
    c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0)
    d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + c_sigma
    c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
    scale = (n + 2.0) / 3.0
    c_1 = min(1.0, scale * 2.0 / ((n + 1.3) ** 2 + mu_eff))
    c_mu = min(
        1.0 - c_1,
        scale * 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff),
    )
    chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

    sigma = cfg.sigma
    diag = np.ones(n)  # the diagonal of C
    p_sigma = np.zeros(n)
    p_c = np.zeros(n)
    rng = np.random.default_rng(cfg.seed)
    run = map_fn or _default_map
    low, high = cfg.weight_range

    result = CMAResult(best_weights=mean.copy(), best_fitness=-math.inf)
    for generation in range(cfg.generations):
        steps = rng.standard_normal((lam, n)) * np.sqrt(diag)
        samples = [np.clip(mean + sigma * step, low, high) for step in steps]
        scores = np.asarray(run(fitness, samples), dtype=float)

        order = np.argsort(-scores)  # maximisation: best first
        if scores[order[0]] > result.best_fitness:
            result.best_fitness = float(scores[order[0]])
            result.best_weights = samples[order[0]].copy()

        old_mean = mean.copy()
        chosen = steps[order[:mu]]
        mean = np.clip(mean + sigma * (weights @ chosen), low, high)

        # step size: is the path of the mean longer than a random walk would be?
        shift = (mean - old_mean) / sigma
        p_sigma = (1 - c_sigma) * p_sigma + math.sqrt(
            c_sigma * (2 - c_sigma) * mu_eff
        ) * shift / np.sqrt(diag)
        norm = float(np.linalg.norm(p_sigma))
        sigma *= math.exp((c_sigma / d_sigma) * (norm / chi_n - 1.0))

        # per-coordinate scale: where did the successful samples actually go?
        h_sigma = (
            norm / math.sqrt(1 - (1 - c_sigma) ** (2 * (generation + 1)))
            < (1.4 + 2.0 / (n + 1)) * chi_n
        )
        p_c = (1 - c_c) * p_c + (math.sqrt(c_c * (2 - c_c) * mu_eff) * shift if h_sigma else 0.0)
        rank_mu = weights @ (chosen**2)
        correction = 0.0 if h_sigma else c_1 * c_c * (2 - c_c) * diag
        diag = (1 - c_1 - c_mu) * diag + c_1 * (p_c**2) + correction + c_mu * rank_mu
        diag = np.maximum(diag, 1e-12)

        stats = CMAGeneration(
            generation=generation,
            best=float(scores.max()),
            mean=float(scores.mean()),
            worst=float(scores.min()),
            sigma=sigma,
        )
        result.history.append(stats)
        if on_generation is not None:
            # the best genome so far, so a caller can checkpoint after every generation
            on_generation(stats, result.best_weights, result.best_fitness)
    return result
