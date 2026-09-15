"""Closed loop environment -> sensory -> brain -> motor -> environment.

Sits below ``learning``: the loop scores ticks through any object with a
``step(...)`` method matching :class:`TickScorer`, so it never imports the
reward module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from doomworm.adapters import SensoryAdapter
from doomworm.brain import Simulator
from doomworm.environments.simple_2d import World


class MotorLike(Protocol):
    """Anything that turns (averaged) neural activity into wheel commands."""

    def __call__(self, activity: Mapping[str, float]) -> tuple[float, float]:
        """Return ``(motor_left, motor_right)``."""
        ...


class TickScorer(Protocol):
    """Anything that turns one tick's events into a reward (see ``learning.reward``)."""

    def step(
        self,
        *,
        x: float,
        y: float,
        ate: bool,
        collided: bool,
        starved: bool,
        reached: bool,
        damaged: bool,
        dead: bool,
        cleaned: int,
        docked: bool,
        battery: float,
    ) -> float:
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
    foods: tuple[tuple[float, float], ...] = ()
    target: tuple[float, float] | None = None
    target_signal: tuple[float, float, float] = (0.0, 0.0, 0.0)
    reached: bool = False
    danger_signal: tuple[float, float, float] = (0.0, 0.0, 0.0)
    health: float = 1.0
    damaged: bool = False
    dead: bool = False
    battery: float = 1.0
    docked: bool = False
    cleaned: int = 0
    coverage: float = 0.0
    activity: dict[str, float] | None = None


def average_activity(window: list[dict[str, float]]) -> dict[str, float]:
    """Mean activity per neuron over a list of per-tick activity dicts."""
    if len(window) == 1:
        return window[0]
    n = float(len(window))
    return {nid: sum(step[nid] for step in window) / n for nid in window[0]}


def run_episode(
    world: World,
    sim: Simulator,
    sensory: SensoryAdapter,
    motor: MotorLike,
    steps: int,
    reward: TickScorer | None = None,
    record_activity: bool = False,
    brain_steps: int = 1,
) -> list[Record]:
    """Step the closed loop until ``steps`` ticks or starvation; return the trace.

    The episode ends early when the agent is dead (starved or out of health).
    ``reward``, when given, scores every tick (Plan §9) and the per-tick value
    lands in :attr:`Record.reward`. ``record_activity`` stores every neuron's
    activity per tick for the debug screen (Plan §41). ``brain_steps`` runs
    that many brain ticks per environment step with the same sensory input
    (Plan §3.1); the motor adapter sees the mean activity over the window.
    """
    if brain_steps < 1:
        raise ValueError("brain_steps must be >= 1")
    trace: list[Record] = []
    obs = world.observe()
    for tick in range(steps):
        currents = sensory(obs.as_channels())
        window = [sim.step(currents) for _ in range(brain_steps)]
        activity = average_activity(window)
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
                reached=obs.reached,
                damaged=obs.damaged,
                dead=world.dead,
                cleaned=obs.cleaned,
                docked=obs.docked,
                battery=obs.battery,
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
                foods=tuple((f.x, f.y) for f in world.foods),
                target=None if world.target is None else (world.target.x, world.target.y),
                target_signal=(obs.target_left, obs.target_front, obs.target_right),
                reached=obs.reached,
                danger_signal=(obs.danger_left, obs.danger_front, obs.danger_right),
                health=obs.health,
                damaged=obs.damaged,
                dead=world.dead,
                activity=dict(activity) if record_activity else None,
                battery=obs.battery,
                docked=obs.docked,
                cleaned=obs.cleaned,
                coverage=world.coverage,
            )
        )
        if world.dead:
            break
    return trace
