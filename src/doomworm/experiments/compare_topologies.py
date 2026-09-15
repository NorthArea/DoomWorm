"""Stage 11: real vs random vs shuffled vs free topology (Plan §17, §49).

Every variant gets the same neurons, sensory/motor mappings, world seeds and
evolution budget. Recorded per variant (Plan §17 "Сравнивать"):

    learning curve (best / mean fitness per generation), generations to a
    positive fitness, held-out fitness, food reached, collisions, survival
    ticks, distance travelled.

Run: ``uv run doomworm compare`` or ``python -m doomworm.experiments.compare_topologies``
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from doomworm.brain import Simulator
from doomworm.connectome import VARIANTS, Connectome, load_cook2019, make_variant
from doomworm.episode import run_episode
from doomworm.experiments.evolve_worm import train_worm
from doomworm.experiments.worm_agent import WormScenario, summarise
from doomworm.learning import EvolutionConfig, RewardTracker

DEFAULT_DIR = Path("runs") / "compare"
FREE_MUTATION_FRACTION = 0.05


@dataclass
class VariantResult:
    """Everything measured for one topology."""

    variant: str
    connections: int
    best_curve: list[float] = field(default_factory=list)
    mean_curve: list[float] = field(default_factory=list)
    train_best: float = math.nan
    gens_to_positive: int | None = None
    held_out_fitness: float = math.nan
    held_out_food: int = 0
    held_out_collisions: int = 0
    held_out_ticks: float = math.nan
    held_out_distance: float = math.nan


def run_variant(
    name: str,
    worm: Connectome,
    config: EvolutionConfig,
    train_seeds: Sequence[int],
    test_seeds: Sequence[int],
    steps: int,
    seed: int,
    out_dir: Path,
    verbose: bool = False,
) -> VariantResult:
    """Train one variant and evaluate it on held-out seeds."""
    connectome = make_variant(worm, name, seed)
    scenario = WormScenario(connectome=connectome)
    cfg = config
    if name == "dense":
        cfg = EvolutionConfig(
            population=config.population,
            generations=config.generations,
            elite_fraction=config.elite_fraction,
            mutation_sigma=config.mutation_sigma,
            weight_range=config.weight_range,
            mutation_fraction=FREE_MUTATION_FRACTION,
        )
    out = out_dir / f"{name}.json"
    result = train_worm(
        scenario,
        cfg,
        train_seeds,
        steps,
        seed,
        out,
        metrics=out.with_suffix(".csv"),
        verbose=verbose,
    )
    vr = VariantResult(variant=name, connections=sum(1 for c in connectome.connections if c.weight))
    vr.best_curve = [g.best for g in result.history]
    vr.mean_curve = [g.mean for g in result.history]
    vr.train_best = result.best_fitness
    vr.gens_to_positive = next((g.generation for g in result.history if g.best > 0), None)

    scenario.template.set_weights(list(result.best_weights))
    totals, ticks, dist = [], [], []
    for s in test_seeds:
        scenario.template.reset()
        world = scenario.make_world(s)
        tracker = RewardTracker()
        trace = run_episode(
            world, Simulator(scenario.template), scenario.sensory, scenario.motor, steps, tracker,
            brain_steps=scenario.brain_steps,
        )  # fmt: skip
        summary = summarise(trace, world)
        totals.append(tracker.total)
        ticks.append(summary["ticks"])
        dist.append(summary["distance"])
        vr.held_out_food += world.food_eaten
        vr.held_out_collisions += world.collisions
    vr.held_out_fitness = float(np.mean(totals))
    vr.held_out_ticks = float(np.mean(ticks))
    vr.held_out_distance = float(np.mean(dist))
    (out_dir / f"{name}_result.json").write_text(json.dumps(asdict(vr), indent=2) + "\n")
    return vr


def summary_table(results: Sequence[VariantResult]) -> str:
    """Markdown table of the comparison."""
    head = (
        "| variant | connections | train best | gens to >0 | held-out fitness | food | "
        "collisions | survival | distance |\n|---|---|---|---|---|---|---|---|---|"
    )
    rows = [
        f"| {r.variant} | {r.connections} | {r.train_best:.1f} | "
        f"{'-' if r.gens_to_positive is None else r.gens_to_positive} | "
        f"{r.held_out_fitness:.1f} | {r.held_out_food} | {r.held_out_collisions} | "
        f"{r.held_out_ticks:.0f} | {r.held_out_distance:.1f} |"
        for r in results
    ]
    return "\n".join([head, *rows])


def plot_curves(results: Sequence[VariantResult], path: Path) -> None:
    """Learning curves, one line per variant."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for r in results:
        ax.plot(r.best_curve, label=f"{r.variant} (best)")
        ax.plot(r.mean_curve, ls="--", alpha=0.5, label=f"{r.variant} (mean)")
    ax.set_xlabel("generation")
    ax.set_ylabel("fitness (mean reward over train seeds)")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(fontsize=8)
    ax.set_title("Plan §17: topology comparison")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def compare(
    variants: Sequence[str],
    config: EvolutionConfig,
    train_seeds: Sequence[int],
    test_seeds: Sequence[int],
    steps: int,
    seed: int,
    out_dir: Path,
    verbose: bool = False,
) -> list[VariantResult]:
    """Run every variant, write summary.md, summary.csv and curves.png."""
    out_dir.mkdir(parents=True, exist_ok=True)
    worm = load_cook2019()
    results = []
    for name in variants:
        if verbose:
            print(f"=== {name} ===", flush=True)
        results.append(
            run_variant(name, worm, config, train_seeds, test_seeds, steps, seed, out_dir, verbose)
        )
    write_summary(results, out_dir)
    return results


def load_results(out_dir: Path, variants: Sequence[str]) -> list[VariantResult]:
    """Read ``<variant>_result.json`` files written by earlier runs."""
    results = []
    for name in variants:
        path = out_dir / f"{name}_result.json"
        if path.exists():
            results.append(VariantResult(**json.loads(path.read_text())))
    return results


def write_summary(results: Sequence[VariantResult], out_dir: Path) -> None:
    """summary.md, summary.csv and curves.png for a set of results."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(summary_table(results) + "\n")
    with (out_dir / "summary.csv").open("w", newline="") as f:
        fields = [k for k in asdict(results[0]) if not k.endswith("_curve")]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: v for k, v in asdict(r).items() if k in fields})
    plot_curves(results, out_dir / "curves.png")


def add_compare_args(parser: argparse.ArgumentParser) -> None:
    """CLI options."""
    parser.add_argument(
        "--summarise",
        action="store_true",
        help="only rebuild summary files from existing <variant>_result.json in --out-dir",
    )
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--train-seeds", type=int, default=3)
    parser.add_argument("--test-seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--sigma", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_DIR)


def run_compare(args: argparse.Namespace) -> int:
    """Entry point for ``doomworm compare``."""
    if args.summarise:
        results = load_results(args.out_dir, args.variants)
        if not results:
            raise SystemExit(f"no <variant>_result.json files in {args.out_dir}")
        write_summary(results, args.out_dir)
        print(summary_table(results))
        return 0
    cfg = EvolutionConfig(
        population=args.population,
        generations=args.generations,
        mutation_sigma=args.sigma,
        weight_range=(-1.0, 1.0),
    )
    results = compare(
        args.variants,
        cfg,
        tuple(range(args.train_seeds)),
        tuple(range(1000, 1000 + args.test_seeds)),
        args.steps,
        args.seed,
        args.out_dir,
        verbose=True,
    )
    print()
    print(summary_table(results))
    print(f"\nsaved {args.out_dir}/summary.md, summary.csv, curves.png and per-variant brains")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Module entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_compare_args(parser)
    return run_compare(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
