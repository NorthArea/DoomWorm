"""Training harness of the A2 bake-off (Plan §20.4, stage 21.2).

One protocol for every trainable candidate: the brain is wrapped in the same
engineered layer as in the benchmark, fitness is the mean benchmark reward on
the training maps, and the search is the stage-4/10 evolution. The candidate
only contributes a weight vector, so the worm, its control topologies and a
network from scratch are trained by exactly the same code.

    doomworm evolve --candidate worm --maps doom4 --task doom --out runs/worm.json
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from multiprocessing import Pool
from multiprocessing.pool import Pool as PoolType
from pathlib import Path
from typing import Any

import numpy as np

from wormlab.candidates.base import Trainable
from wormlab.candidates.registry import CandidateSpec, build_candidate
from wormlab.episode import BrainLike
from wormlab.learning.benchmark import BenchmarkConfig, run_benchmark
from wormlab.learning.evolution import EvolutionConfig, EvolutionResult, GenerationStats, evolve

LAYERS = ("none", "memory")


@dataclass(frozen=True)
class TrainConfig:
    """Where and how long a candidate trains; the world is the benchmark's world."""

    layer: str = "none"
    train_seeds: tuple[int, ...] = (100, 101, 102)
    train_repeats: int = 1  # sensor-noise seeds per training map
    steps: int = 800
    population: int = 40
    generations: int = 25
    sigma: float = 0.02
    init_sigma: float | None = None
    weight_range: tuple[float, float] = (-1.0, 1.0)
    mutation_fraction: float = 1.0
    seed: int = 0
    workers: int = 1
    search: str = "mutation"  # "mutation" (stage 4) or "cma" (sep-CMA-ES, stage 18)

    def evolution(self) -> EvolutionConfig:
        """The search hyper-parameters."""
        return EvolutionConfig(
            population=self.population,
            generations=self.generations,
            mutation_sigma=self.sigma,
            weight_range=self.weight_range,
            mutation_fraction=self.mutation_fraction,
        )


def wrap(brain: BrainLike, spec: CandidateSpec, layer: str) -> BrainLike:
    """The condition a candidate is trained and benchmarked under (Plan §9)."""
    if layer not in LAYERS:
        raise ValueError(f"layer must be one of {LAYERS}")
    if layer == "memory":
        from wormlab.layer import MemoryLayer

        return MemoryLayer(brain)
    return brain


def benchmark_config(spec: CandidateSpec, cfg: TrainConfig) -> BenchmarkConfig:
    """Benchmark settings for the training maps (``train_repeats`` noise seeds per map)."""
    return BenchmarkConfig(
        maps=spec.maps,
        task=spec.task,
        dangers=spec.dangers,
        sensors=spec.sensors,
        test_seeds=cfg.train_seeds,
        steps=cfg.steps,
        repeats=cfg.train_repeats,
    )


_CACHE: dict[tuple[CandidateSpec, str], tuple[Trainable, BrainLike]] = {}


def fitness_of(spec: CandidateSpec, cfg: TrainConfig, weights: np.ndarray) -> float:
    """Mean benchmark reward of ``weights`` on the training maps (deterministic).

    Module-level and picklable so a worker pool can evaluate a generation; each
    process builds the candidate once and keeps it.
    """
    key = (spec, cfg.layer)
    if key not in _CACHE:
        brain = build_candidate(spec)
        _CACHE[key] = (brain, wrap(brain, spec, cfg.layer))
    trainable, wrapped = _CACHE[key]
    trainable.set_weights([float(w) for w in weights])
    result = run_benchmark(wrapped, spec.label(), benchmark_config(spec, cfg))
    return result.mean("reward")


def _pool_map(
    pool: PoolType, fn: Callable[[np.ndarray], float], genomes: Iterable[np.ndarray]
) -> list[float]:
    return [float(v) for v in pool.map(fn, list(genomes))]


def _run_cma(
    fitness: Callable[[np.ndarray], float],
    initial: np.ndarray,
    cfg: TrainConfig,
    record: Callable[[GenerationStats, np.ndarray, float], None],
    map_fn: Callable[..., Sequence[float]] | None,
) -> EvolutionResult:
    """The same harness, driven by sep-CMA-ES instead of fixed-sigma mutation."""
    from wormlab.learning.cmaes import CMAConfig, cma_es

    def on_generation(stats: Any, best: np.ndarray, best_fitness: float) -> None:
        record(
            GenerationStats(
                generation=stats.generation,
                best=stats.best,
                mean=stats.mean,
                worst=stats.worst,
            ),
            best,
            best_fitness,
        )

    result = cma_es(
        fitness,
        initial,
        CMAConfig(
            generations=cfg.generations,
            population=cfg.population,
            sigma=cfg.init_sigma or 0.2,
            weight_range=cfg.weight_range,
            seed=cfg.seed,
        ),
        on_generation=on_generation,
        map_fn=map_fn,
    )
    return EvolutionResult(
        best_weights=result.best_weights,
        best_fitness=result.best_fitness,
        history=[
            GenerationStats(generation=g.generation, best=g.best, mean=g.mean, worst=g.worst)
            for g in result.history
        ],
    )


def train_candidate(
    spec: CandidateSpec,
    cfg: TrainConfig,
    out: Path,
    on_generation: Callable[[GenerationStats], None] | None = None,
) -> EvolutionResult:
    """Evolve one candidate in place and save brain JSON + per-generation CSV."""
    brain = build_candidate(spec)
    initial = np.asarray(brain.get_weights(), dtype=float)
    fitness = partial(fitness_of, spec, cfg)
    history: list[GenerationStats] = []

    out.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint(stats: GenerationStats, best: np.ndarray, best_fitness: float) -> None:
        # Save the best genome so far after every generation: a killed run keeps its progress.
        brain.set_weights([float(w) for w in best])
        brain.meta.update(
            {
                "candidate": spec.kind,
                "layer": cfg.layer,
                "train": asdict(cfg) | {"spec": asdict(spec)},
                "fitness": best_fitness,
                "generations": stats.generation + 1,
            }
        )
        brain.save(out)

    def record(stats: GenerationStats, best: np.ndarray, best_fitness: float) -> None:
        history.append(stats)
        checkpoint(stats, best, best_fitness)
        with out.with_suffix(".csv").open("a", newline="") as f:
            csv.writer(f).writerow([stats.generation, stats.best, stats.mean, stats.worst])
        if on_generation is not None:
            on_generation(stats)

    with out.with_suffix(".csv").open("w", newline="") as f:
        csv.writer(f).writerow(["generation", "best", "mean", "worst"])

    kwargs: dict[str, Any] = {}
    if cfg.workers > 1:
        pool = Pool(cfg.workers)
        kwargs["map_fn"] = partial(_pool_map, pool)
    try:
        if cfg.search == "cma":
            result = _run_cma(fitness, initial, cfg, record, kwargs.get("map_fn"))
        else:
            result = evolve(
                fitness,
                brain.n_weights,
                cfg.evolution(),
                seed=cfg.seed,
                on_generation_best=record,
                initial=initial,
                initial_sigma=cfg.init_sigma,
                **kwargs,
            )
    finally:
        if cfg.workers > 1:
            pool.close()
            pool.join()

    checkpoint(result.history[-1], result.best_weights, result.best_fitness)
    out.with_suffix(".meta.json").write_text(json.dumps(brain.meta, indent=2) + "\n")
    return result
