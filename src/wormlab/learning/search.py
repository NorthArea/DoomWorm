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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

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
    consensus: int = 1
    """How many of the best branches to average into the decision (stage 23b).

    One takes the single luckiest draw, which is what the first version did and
    which turned out not to be a function of the state at all: the same state
    gave a different answer every time, so there was nothing to imitate. The
    mean of the best k averages the noise out and leaves whatever the state
    actually implies — the part a brain could learn.
    """


@dataclass
class SearchRow:
    """What one decision looked like, for the competence map of stage 22."""

    tick: int
    scores: list[float]
    chosen: int
    chosen_drive: Drive
    modal_drive: Drive  # what the brain would have done on its own, unsearched
    channels: dict[str, float] = field(default_factory=dict)


def _mean_drive(drives: Sequence[Drive]) -> Drive:
    """The average intent of several branches (stage 23b's consensus)."""
    n = float(len(drives))
    return Drive(
        forward=sum(d.forward for d in drives) / n,
        turn=sum(d.turn for d in drives) / n,
        strafe=sum(d.strafe for d in drives) / n,
        fire=sum(d.fire for d in drives) / n,
    )


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


def rollout_branches(
    world: World,
    brain: BrainLike,
    candidates: int,
    horizon: int,
    reward: RewardConfig | None = None,
    sensors: SensorSuite | None = None,
) -> tuple[list[float], list[Drive]]:
    """Draw `candidates` proposals from *one* state and score each in its own copy.

    The neural snapshot is restored before every branch and once more at the
    end. Without that, branch k starts from wherever branch k-1's rollout left
    the neurons: the proposals stop being draws from a single state, and any
    measurement of what that state implies measures the drift instead. This
    function exists because two probes reimplemented the loop and left it out.
    """
    reward_config = reward or RewardConfig()
    state = brain.sim.snapshot()  # type: ignore[attr-defined]
    scores: list[float] = []
    firsts: list[Drive] = []
    for _ in range(candidates):
        brain.sim.restore(state)  # type: ignore[attr-defined]
        score, first = _score_branch(copy.deepcopy(world), brain, horizon, reward_config, sensors)
        scores.append(score)
        firsts.append(first)
    brain.sim.restore(state)  # type: ignore[attr-defined]
    return scores, firsts


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
            scores, firsts = rollout_branches(
                world, brain, cfg.candidates, cfg.horizon, reward_config, sensors
            )
            order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
            best = order[0]
            top = order[: max(1, min(cfg.consensus, len(order)))]
            held = _mean_drive([firsts[i] for i in top])
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


class RandomProposer:
    """The control every proposal search has to beat: intents drawn from nothing.

    Stage 23 found that the rollout over *these* scores as well as the rollout
    over the connectome's own proposals, which is why this lives in the library
    and not in a scratch script. A search is measured against it or not at all.

    It carries a `sim` with the same snapshot/restore contract as a real brain,
    so :func:`searched_episode` cannot tell the difference.
    """

    def __init__(self, seed: int = 0) -> None:
        self.sim = self  # the search snapshots the brain's state; ours is the RNG's
        self.rng = np.random.default_rng(seed)
        self.fire = 0.0

    def reset(self) -> None:
        """Nothing to forget."""

    def snapshot(self) -> Any:
        """Nothing: a real brain's snapshot holds its voltages, never its noise.

        Restoring the RNG here would make every branch an identical draw and
        quietly turn the control into a constant.
        """
        return None

    def restore(self, state: Any) -> None:
        """Nothing to put back."""
        del state

    def act(self, channels: Mapping[str, float]) -> tuple[float, float]:
        """A wheel pair with no relation to the channels at all."""
        del channels
        self.fire = float(self.rng.random())
        return float(self.rng.uniform(-1, 1)), float(self.rng.uniform(-1, 1))
