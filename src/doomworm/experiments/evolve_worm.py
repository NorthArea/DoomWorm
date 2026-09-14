"""Stage 10: train the synaptic weights of the C. elegans connectome (Plan §16).

Topology is fixed. The genome is the 5905 weights of the stage-9 network,
initialised from the dataset (stage 5) and perturbed; evolution may change
magnitudes and signs. Fitness = mean total reward over seeded worlds.

Run: ``uv run doomworm train --scenario worm`` or ``python -m doomworm.experiments.evolve_worm``
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from doomworm.brain import save_brain
from doomworm.experiments.worm_agent import SCENARIO_NAME, WormScenario
from doomworm.learning import EvolutionConfig, EvolutionResult, GenerationStats, evaluate, evolve

DEFAULT_OUT = Path("runs") / "worm_evolved.json"


def train_worm(
    scenario: WormScenario,
    config: EvolutionConfig,
    train_seeds: Sequence[int],
    steps: int,
    seed: int,
    out: Path,
    initial_sigma: float | None = None,
    metrics: Path | None = None,
    verbose: bool = False,
) -> EvolutionResult:
    """Evolve the connectome weights from their dataset initialisation; save the best."""
    initial = np.array(scenario.template.get_weights())
    rows: list[dict[str, float]] = []

    def fitness(weights: np.ndarray) -> float:
        return evaluate(scenario, weights, train_seeds, steps).fitness

    def report(stats: GenerationStats) -> None:
        rows.append({"generation": stats.generation, "best": stats.best, "mean": stats.mean})
        if verbose:
            print(
                f"gen {stats.generation:>3}  best {stats.best:>8.2f}  "
                f"mean {stats.mean:>8.2f}  worst {stats.worst:>8.2f}",
                flush=True,
            )

    result = evolve(
        fitness,
        scenario.n_weights,
        config,
        seed=seed,
        on_generation=report,
        initial=initial,
        initial_sigma=initial_sigma,
    )
    scenario.template.set_weights(list(result.best_weights))
    save_brain(
        out,
        scenario.template,
        meta={
            "stage": 10,
            "scenario": SCENARIO_NAME,
            "params": scenario.params,
            "seed": seed,
            "train_seeds": list(train_seeds),
            "steps": steps,
            "generations": config.generations,
            "population": config.population,
            "mutation_sigma": config.mutation_sigma,
            "fitness": result.best_fitness,
        },
    )
    if metrics is not None:
        metrics.parent.mkdir(parents=True, exist_ok=True)
        with metrics.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["generation", "best", "mean"])
            writer.writeheader()
            writer.writerows(rows)
    return result


def add_worm_args(parser: argparse.ArgumentParser) -> None:
    """Worm-specific training options."""
    parser.add_argument("--sigma", type=float, default=0.02, help="mutation sigma (weights)")
    parser.add_argument("--init-sigma", type=float, default=None, help="first-generation spread")


def run_train(args: argparse.Namespace) -> int:
    """Train from parsed CLI args and report held-out results."""
    scenario = WormScenario()
    cfg = EvolutionConfig(
        population=args.population,
        generations=args.generations,
        mutation_sigma=args.sigma,
        weight_range=(-1.0, 1.0),
    )
    train_seeds = tuple(range(args.train_seeds))
    out = args.out if args.out != Path("runs") / "small_evolved.json" else DEFAULT_OUT
    metrics = out.with_suffix(".csv")
    baseline = evaluate(scenario, scenario.template.get_weights(), train_seeds, args.steps)
    print(
        f"untrained worm on train seeds: fitness {baseline.fitness:.2f}, food {baseline.food_eaten}"
    )
    result = train_worm(
        scenario, cfg, train_seeds, args.steps, args.seed, out,
        initial_sigma=args.init_sigma, metrics=metrics, verbose=True,
    )  # fmt: skip

    test_seeds = tuple(range(1000, 1004))
    held_out = evaluate(scenario, result.best_weights, test_seeds, args.steps)
    scenario.template.set_weights(list(np.array(scenario.template.get_weights())))
    print(f"\nbest train fitness: {result.best_fitness:.2f} (untrained {baseline.fitness:.2f})")
    print(
        f"held-out seeds {test_seeds}: fitness {held_out.fitness:.2f}, "
        f"food {held_out.food_eaten}, collisions {held_out.collisions}"
    )
    print(f"saved {out} and {metrics}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Module entry point."""
    from doomworm.experiments.evolve_small import add_train_args

    parser = argparse.ArgumentParser(description=__doc__)
    add_train_args(parser)
    add_worm_args(parser)
    return run_train(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
