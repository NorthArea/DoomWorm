"""Stage 4: evolve the weights of the stage-2 topology from random initial values.

Topology, thresholds and adapter wiring are exactly those of ``food_agent``;
only the 12 synaptic weights are searched. Each genome is scored on several
seeded worlds (random start pose, random food with respawn) and fitness is
the mean total reward (Plan §9-10).

Run: ``uv run doomworm train`` or ``python -m doomworm.experiments.evolve_small``
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from doomworm.brain import Network, save_brain
from doomworm.environments.simple_2d import AgentState, Obstacle, World
from doomworm.experiments import food_agent, obstacle_agent
from doomworm.learning import EvolutionConfig, EvolutionResult, GenerationStats, evaluate, evolve

SCENARIO_NAME = "small_food"
DEFAULT_OUT = Path("runs") / "small_evolved.json"
N_FOOD = 2


class SmallFoodScenario:
    """Stage-2 brain topology in a 20x20 world with one obstacle and respawning food."""

    def __init__(self) -> None:
        self.template: Network = food_agent.extend_brain(obstacle_agent.build_brain())
        _, _, self.sensory, self.motor = food_agent.build_scenario()
        self.brain_steps = 1

    @property
    def n_weights(self) -> int:
        """Genome length."""
        return len(self.template.synapses)

    def random_weights(self, seed: int) -> np.ndarray:
        """A random genome, same distribution as the initial population."""
        return np.random.default_rng(seed).uniform(-1.0, 1.0, size=self.n_weights)

    def make_world(self, seed: int) -> World:
        """Fixed obstacle; seeded start pose and food positions; food respawns."""
        world = World(
            width=20.0,
            height=20.0,
            obstacles=[Obstacle(x=10.0, y=10.0, radius=1.5)],
            respawn_food=True,
            seed=seed,
        )
        rng = world.rng
        while True:
            pose = AgentState(
                x=rng.uniform(1.0, 19.0),
                y=rng.uniform(1.0, 19.0),
                heading=rng.uniform(-math.pi, math.pi),
            )
            if all(
                math.dist((pose.x, pose.y), (o.x, o.y)) > o.radius + 1.0 for o in world.obstacles
            ):
                break
        world.agent = pose
        world.foods = [world.spawn_food() for _ in range(N_FOOD)]
        return world


def train(
    config: EvolutionConfig,
    train_seeds: Sequence[int],
    steps: int,
    seed: int,
    out: Path,
    verbose: bool = False,
) -> EvolutionResult:
    """Evolve weights, save the best brain as JSON, return the result."""
    scenario = SmallFoodScenario()

    def fitness(weights: np.ndarray) -> float:
        return evaluate(scenario, weights, train_seeds, steps).fitness

    def report(stats: GenerationStats) -> None:
        if verbose:
            print(
                f"gen {stats.generation:>3}  best {stats.best:>8.2f}  "
                f"mean {stats.mean:>8.2f}  worst {stats.worst:>8.2f}"
            )

    result = evolve(fitness, scenario.n_weights, config, seed=seed, on_generation=report)
    scenario.template.set_weights(list(result.best_weights))
    save_brain(
        out,
        scenario.template,
        meta={
            "stage": 4,
            "scenario": SCENARIO_NAME,
            "seed": seed,
            "train_seeds": list(train_seeds),
            "steps": steps,
            "generations": config.generations,
            "population": config.population,
            "fitness": result.best_fitness,
        },
    )
    return result


def add_train_args(parser: argparse.ArgumentParser) -> None:
    """CLI options shared by ``doomworm train`` and the module entry point."""
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--population", type=int, default=60)
    parser.add_argument("--train-seeds", type=int, default=4, help="number of seeded maps")
    parser.add_argument("--steps", type=int, default=600, help="max ticks per episode")
    parser.add_argument("--seed", type=int, default=0, help="evolution RNG seed")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)


def run_train(args: argparse.Namespace) -> int:
    """Train from parsed CLI args, print a held-out evaluation."""
    cfg = EvolutionConfig(population=args.population, generations=args.generations)
    train_seeds = tuple(range(args.train_seeds))
    result = train(cfg, train_seeds, args.steps, args.seed, args.out, verbose=True)

    scenario = SmallFoodScenario()
    test_seeds = tuple(range(1000, 1004))
    held_out = evaluate(scenario, result.best_weights, test_seeds, args.steps)
    print(f"\nbest train fitness: {result.best_fitness:.2f}")
    print(
        f"held-out seeds {test_seeds}: fitness {held_out.fitness:.2f}, "
        f"food {held_out.food_eaten}, collisions {held_out.collisions}"
    )
    print(f"saved {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Module entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_train_args(parser)
    return run_train(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
