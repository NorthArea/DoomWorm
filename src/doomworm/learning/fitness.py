"""Fitness = mean total reward over several seeded episodes (Plan §10)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from doomworm.adapters import SensoryAdapter
from doomworm.brain import Network, Simulator
from doomworm.environments.simple_2d import World
from doomworm.episode import MotorLike, run_episode
from doomworm.learning.reward import RewardConfig, RewardTracker


class Scenario(Protocol):
    """A fixed topology plus a seeded world factory and adapters."""

    template: Network
    sensory: SensoryAdapter
    brain_steps: int

    @property
    def motor(self) -> MotorLike:
        """Activity -> wheels."""
        ...

    def make_world(self, seed: int) -> World:
        """Build a fresh, seeded world."""
        ...


@dataclass(frozen=True)
class Evaluation:
    """Aggregate outcome of a genome over several seeds."""

    fitness: float
    food_eaten: int
    collisions: int
    ticks: int


def evaluate(
    scenario: Scenario,
    weights: Sequence[float] | np.ndarray,
    seeds: Sequence[int],
    steps: int,
    reward_config: RewardConfig | None = None,
) -> Evaluation:
    """Run one episode per seed with ``weights`` loaded into the scenario topology."""
    net = scenario.template
    net.set_weights([float(w) for w in weights])
    totals: list[float] = []
    food = collisions = ticks = 0
    for seed in seeds:
        net.reset()
        world = scenario.make_world(seed)
        tracker = RewardTracker(reward_config or RewardConfig())
        trace = run_episode(
            world,
            Simulator(net),
            scenario.sensory,
            scenario.motor,
            steps,
            tracker,
            brain_steps=scenario.brain_steps,
        )
        totals.append(tracker.total)
        food += world.food_eaten
        collisions += world.collisions
        ticks += len(trace)
    return Evaluation(
        fitness=float(np.mean(totals)), food_eaten=food, collisions=collisions, ticks=ticks
    )
