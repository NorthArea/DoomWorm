"""Closed loop environment -> sensory -> brain -> motor -> environment.

Sits below ``learning``: the loop scores ticks through any object with a
``step(...)`` method matching :class:`TickScorer`, so it never imports the
reward module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from doomworm.adapters import MotorAdapter, SensoryAdapter
from doomworm.brain import Simulator
from doomworm.environments.simple_2d import World


class TickScorer(Protocol):
    """Anything that turns one tick's events into a reward (see ``learning.reward``)."""

    def step(self, *, x: float, y: float, ate: bool, collided: bool, starved: bool) -> float:
        """Score one tick."""
        ...


@dataclass(frozen=True)
class Record:
    """One tick of an episode."""

    tick: int
    x: float
    y: float
    heading: float
    obstacle: tuple[float, float, float]
    food: tuple[float, float, float]
    hunger: float
    motors: tuple[float, float]
    collided: bool
    ate: bool
    starved: bool = False
    reward: float = 0.0


def run_episode(
    world: World,
    sim: Simulator,
    sensory: SensoryAdapter,
    motor: MotorAdapter,
    steps: int,
    reward: TickScorer | None = None,
) -> list[Record]:
    """Step the closed loop until ``steps`` ticks or starvation; return the trace.

    ``reward``, when given, scores every tick (Plan §9) and the per-tick value
    lands in :attr:`Record.reward`.
    """
    trace: list[Record] = []
    obs = world.observe()
    for tick in range(steps):
        activity = sim.step(sensory(obs.as_channels()))
        motors = motor(activity)
        obs = world.step(*motors)
        tick_reward = 0.0
        if reward is not None:
            tick_reward = reward.step(
                x=world.agent.x,
                y=world.agent.y,
                ate=obs.ate,
                collided=obs.collided,
                starved=world.starved,
            )
        trace.append(
            Record(
                tick=tick,
                x=world.agent.x,
                y=world.agent.y,
                heading=world.agent.heading,
                obstacle=(obs.sensor_left, obs.sensor_front, obs.sensor_right),
                food=(obs.food_left, obs.food_front, obs.food_right),
                hunger=obs.hunger,
                motors=motors,
                collided=obs.collided,
                ate=obs.ate,
                starved=world.starved,
                reward=tick_reward,
            )
        )
        if world.starved:
            break
    return trace
