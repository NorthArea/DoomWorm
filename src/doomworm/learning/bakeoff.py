"""Training harness of the A2 bake-off (Plan §20.4, stage 21.2).

One protocol for every trainable candidate: the brain is wrapped in the same
engineered layer as in the benchmark, fitness is the mean benchmark reward on
the training maps, and the search is the stage-4/10 evolution. The candidate
only contributes a weight vector, so the worm, its control topologies and a
network from scratch are trained by exactly the same code.

    doomworm evolve --candidate worm --layer needs --out runs/a2/worm.json
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from functools import partial
from multiprocessing import Pool
from multiprocessing.pool import Pool as PoolType
from pathlib import Path
from typing import Any

import numpy as np

from doomworm.brains.base import Trainable
from doomworm.brains.candidates import CandidateSpec, build_candidate
from doomworm.brains.planner_layer import PlannerLayer
from doomworm.environments.sensors import PRESETS
from doomworm.episode import BrainLike
from doomworm.learning.benchmark import BenchmarkConfig, run_benchmark
from doomworm.learning.evolution import EvolutionConfig, EvolutionResult, GenerationStats, evolve

LAYERS = ("none", "coverage", "needs")


@dataclass(frozen=True)
class TrainConfig:
    """Where and how long a candidate trains; the world is the benchmark's world."""

    layer: str = "needs"
    train_seeds: tuple[int, ...] = (100, 101, 102)
    steps: int = 800
    population: int = 40
    generations: int = 25
    sigma: float = 0.02
    init_sigma: float | None = None
    weight_range: tuple[float, float] = (-1.0, 1.0)
    mutation_fraction: float = 1.0
    seed: int = 0
    workers: int = 1

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
    """The engineered layer every candidate gets in the benchmark (none = bare)."""
    if layer not in LAYERS:
        raise ValueError(f"layer must be one of {LAYERS}")
    if layer == "none":
        return brain
    return PlannerLayer(brain, PRESETS[spec.sensors], mode=layer)


def benchmark_config(spec: CandidateSpec, cfg: TrainConfig) -> BenchmarkConfig:
    """Benchmark settings for the training maps: one noise repeat per map."""
    return BenchmarkConfig(
        maps=spec.maps,
        task=spec.task,
        dangers=spec.dangers,
        sensors=spec.sensors,
        test_seeds=cfg.train_seeds,
        steps=cfg.steps,
        repeats=1,
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

    def record(stats: GenerationStats) -> None:
        history.append(stats)
        if on_generation is not None:
            on_generation(stats)

    kwargs: dict[str, Any] = {}
    if cfg.workers > 1:
        pool = Pool(cfg.workers)
        kwargs["map_fn"] = partial(_pool_map, pool)
    try:
        result = evolve(
            fitness,
            brain.n_weights,
            cfg.evolution(),
            seed=cfg.seed,
            on_generation=record,
            initial=initial,
            initial_sigma=cfg.init_sigma,
            **kwargs,
        )
    finally:
        if cfg.workers > 1:
            pool.close()
            pool.join()

    brain.set_weights([float(w) for w in result.best_weights])
    brain.meta.update(
        {
            "candidate": spec.kind,
            "layer": cfg.layer,
            "train": asdict(cfg) | {"spec": asdict(spec)},
            "fitness": result.best_fitness,
            "generations": len(result.history),
        }
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    brain.save(out)
    with out.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "best", "mean", "worst"])
        for g in result.history:
            writer.writerow([g.generation, g.best, g.mean, g.worst])
    out.with_suffix(".meta.json").write_text(json.dumps(brain.meta, indent=2) + "\n")
    return result
