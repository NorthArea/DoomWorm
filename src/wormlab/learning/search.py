"""Stage 21: search over the brain's own proposals.

With noise in the neurons the brain stops being a function and becomes a
distribution: run the same tick from the same state twice and it answers twice
differently. That makes a search possible that has nothing to do with training.

    from one state, draw N intents from the connectome
    roll each of them forward H ticks in a *copy* of the world
    keep the first action of whichever branch scored best

The connectome is the proposal distribution and the rollout is the judge. This
is not a brain that plays better — it is a measurement of how good the worm's
proposals are, and the first question it answers is whether searching over them
helps at all. If it does not, there is nothing worth teaching back (stage 23).

Two honest limits. The rollout uses the real simulator, so the system is given
something the animal has no trace of — the ability to look ahead — and that is
a declared condition, not a property of the worm. And the engine levels cannot
be searched at all: a ViZDoom process has no copy.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field

from wormlab.body import Drive, drive_of
from wormlab.environments.sensors import SensorSuite
from wormlab.environments.simple_2d import World
from wormlab.episode import BrainLike
from wormlab.learning.reward import RewardConfig, RewardTracker


@dataclass(frozen=True)
class SearchConfig:
    """How wide and how far the search looks."""

    candidates: int = 8  # intents drawn from the brain per decision
    horizon: int = 15  # ticks each one is rolled forward
    plan_every: int = 5  # decisions are held for this many ticks (standard MPC)
    noise: float = 0.3  # the brain must be stochastic or every candidate is the same


@dataclass
class SearchRow:
    """What one decision looked like, for the competence map of stage 22."""

    tick: int
    scores: list[float]
    chosen: int
    chosen_drive: Drive
    modal_drive: Drive  # what the brain would have done on its own, unsearched
    channels: dict[str, float] = field(default_factory=dict)


def _score_branch(
    world: World,
    brain: BrainLike,
    horizon: int,
    reward: RewardConfig,
    sensors: SensorSuite | None,
) -> tuple[float, Drive]:
    """Roll one branch forward in a copy of the world; return its reward and first act."""
    tracker = RewardTracker(reward)
    total = 0.0
    first: Drive | None = None
    obs = world.observe()
    for _ in range(horizon):
        channels = obs.as_channels() if sensors is None else sensors.read(world, obs)
        intent = drive_of(brain.act(channels), float(getattr(brain, "fire", 0.0)))
        if first is None:
            first = intent
        obs = world.drive(intent)
        total += tracker.step(
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
            hit=obs.hit,
            killed=obs.killed,
        )
        if world.finished:
            break
    return total, first or Drive()


def searched_episode(
    world: World,
    brain: BrainLike,
    steps: int,
    config: SearchConfig | None = None,
    reward: RewardConfig | None = None,
    sensors: SensorSuite | None = None,
    on_decision: Callable[[SearchRow], None] | None = None,
) -> tuple[float, list[SearchRow]]:
    """Play an episode where every decision is searched; return reward and the log."""
    cfg = config or SearchConfig()
    reward_config = reward or RewardConfig()
    tracker = RewardTracker(reward_config)
    rows: list[SearchRow] = []
    brain.reset()
    if sensors is not None:
        sensors.reset(world)
    obs = world.observe()
    held = Drive()
    total = 0.0

    for tick in range(steps):
        channels = obs.as_channels() if sensors is None else sensors.read(world, obs)
        if tick % cfg.plan_every == 0:
            state = brain.sim.snapshot()  # type: ignore[attr-defined]
            scores: list[float] = []
            firsts: list[Drive] = []
            for _ in range(cfg.candidates):
                brain.sim.restore(state)  # type: ignore[attr-defined]
                score, first = _score_branch(
                    copy.deepcopy(world), brain, cfg.horizon, reward_config, sensors
                )
                scores.append(score)
                firsts.append(first)
            brain.sim.restore(state)  # type: ignore[attr-defined]
            best = max(range(len(scores)), key=scores.__getitem__)
            held = firsts[best]
            row = SearchRow(
                tick=tick,
                scores=scores,
                chosen=best,
                chosen_drive=held,
                modal_drive=firsts[0],  # the first draw: what it would have done unsearched
                channels=dict(channels),
            )
            rows.append(row)
            if on_decision is not None:
                on_decision(row)

        obs = world.drive(held)
        total += tracker.step(
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
            hit=obs.hit,
            killed=obs.killed,
        )
        if world.finished:
            break
    return total, rows
