"""Stage 3 demo: the same world scored for three hand-wired brains.

    blind     no sensors wired; drives straight into the wall and starves
    obstacle  stage-1 brain: avoids walls, ignores food, starves
    food      stage-2 brain: avoids walls, eats when hungry

Run: ``uv run python -m wormlab.experiments.reward_demo [--steps N]``
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from wormlab.adapters import MotorAdapter, SensoryAdapter
from wormlab.brain import Network, Neuron, Simulator
from wormlab.experiments import food_agent, obstacle_agent
from wormlab.experiments.episode import Record, run_episode
from wormlab.learning import RewardConfig, RewardTracker


@dataclass(frozen=True)
class Result:
    """Outcome of one scored episode."""

    name: str
    total: float
    breakdown: dict[str, float]
    ticks: int
    food_eaten: int
    collisions: int
    cells: int
    trace: list[Record]


def build_blind_scenario() -> tuple[Simulator, SensoryAdapter, MotorAdapter]:
    """Motor neurons on tonic drive, nothing else."""
    net = Network()
    for nid in obstacle_agent.MOTOR_NEURONS:
        net.add_neuron(Neuron(nid, threshold=1.0, decay=1.0))
    sensory = SensoryAdapter(channels={}, tonic=dict.fromkeys(obstacle_agent.MOTOR_NEURONS, 1.0))
    return Simulator(net), sensory, MotorAdapter(left="M_LEFT", right="M_RIGHT")


def evaluate_all(steps: int, config: RewardConfig | None = None) -> list[Result]:
    """Score blind, obstacle and food brains on the stage-2 world."""
    config = config or RewardConfig()
    brains = {
        "blind": build_blind_scenario(),
        "obstacle": obstacle_agent.build_scenario()[1:],
        "food": food_agent.build_scenario()[1:],
    }
    results = []
    for name, (sim, sensory, motor) in brains.items():
        world = food_agent.build_world()
        tracker = RewardTracker(config)
        trace = run_episode(world, sim, sensory, motor, steps, reward=tracker)
        results.append(
            Result(
                name=name,
                total=tracker.total,
                breakdown=dict(tracker.breakdown),
                ticks=len(trace),
                food_eaten=world.food_eaten,
                collisions=world.collisions,
                cells=tracker.cells_visited,
                trace=trace,
            )
        )
    return results


def main(argv: list[str] | None = None) -> int:
    """Print a comparison table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args(argv)

    results = evaluate_all(args.steps)
    print(
        f"{'brain':<9}{'total':>8} | {'food':>6} {'coll':>7} {'explore':>8} {'starve':>7} |"
        f" {'ticks':>5} {'ate':>4} {'hits':>5} {'cells':>6}"
    )
    for r in results:
        b = r.breakdown
        print(
            f"{r.name:<9}{r.total:>8.1f} | {b['food']:>6.1f} {b['collision']:>7.1f} "
            f"{b['explore']:>8.1f} {b['starvation']:>7.1f} | {r.ticks:>5} {r.food_eaten:>4} "
            f"{r.collisions:>5} {r.cells:>6}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
