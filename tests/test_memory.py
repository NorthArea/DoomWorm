"""Stage 16: a place memory as a layer, not a planner."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import pytest

from doomworm.candidates import RNNBrain, ScriptedBrain
from doomworm.episode import BrainLike
from doomworm.layer import NOVELTY, MemoryLayer


class Recorder:
    """A brain that drives straight and remembers what it was told."""

    name = "recorder"

    def __init__(self) -> None:
        self.seen: list[dict[str, float]] = []

    def reset(self) -> None:
        self.seen.clear()

    def act(self, channels: Mapping[str, float]) -> tuple[float, float]:
        self.seen.append(dict(channels))
        return 1.0, 1.0


def test_the_layer_adds_channels_and_never_touches_the_output() -> None:
    inner = ScriptedBrain(lambda _c: (0.3, -0.7), "fixed")
    layer = MemoryLayer(inner)
    layer.reset()
    assert layer.act({"odom_x": 0.0, "odom_y": 0.0, "odom_heading": 0.0}) == (0.3, -0.7)
    assert set(layer.last_novelty) == set(NOVELTY)


def test_unvisited_ground_reads_as_a_gradient_in_its_sector() -> None:
    inner = Recorder()
    layer = MemoryLayer(inner, cell=1.0, radius=6.0)
    layer.reset()
    # facing +x from the origin: everything is unvisited, so the nearest cell is ahead
    layer.act({"odom_x": 0.5, "odom_y": 0.5, "odom_heading": 0.0})
    first = inner.seen[-1]
    assert sum(first[k] for k in NOVELTY) > 0.0, "somewhere is unexplored"

    # walk east, marking cells; then face west, where we came from
    for step in range(6):
        layer.act({"odom_x": 0.5 + step, "odom_y": 0.5, "odom_heading": 0.0})
    layer.act({"odom_x": 5.5, "odom_y": 0.5, "odom_heading": math.pi})
    behind = inner.seen[-1]
    assert behind["novelty_front"] == 0.0, "the way back is all visited"
    assert behind["novelty_left"] + behind["novelty_right"] > 0.0, "the sides are not"


def test_memory_is_per_episode_and_can_be_made_to_forget() -> None:
    layer = MemoryLayer(Recorder(), cell=1.0)
    layer.reset()
    for step in range(5):
        layer.act({"odom_x": 0.5 + step, "odom_y": 0.5, "odom_heading": 0.0})
    assert layer.explored == 5
    layer.reset()
    assert layer.explored == 0

    forgetful = MemoryLayer(Recorder(), cell=1.0, forget=3)
    forgetful.reset()
    for step in range(5):
        forgetful.act({"odom_x": 0.5 + step, "odom_y": 0.5, "odom_heading": 0.0})
    # five cells marked at ticks 0..4, the clock now reads 5, and a cell is fresh
    # while tick - seen < forget: ticks 3 and 4 survive, the rest are new ground again
    assert forgetful.explored == 2, "older cells count as unvisited again"


def test_the_layer_reaches_the_worm_and_only_through_the_channels() -> None:
    """The worm reads novelty (stage 16); a brain without that route cannot tell."""
    from doomworm.candidates import CandidateSpec, WormBrain, build_candidate
    from doomworm.connectome import default_sensory_mapping
    from doomworm.environments.worlds import build_world
    from doomworm.episode import run_brain_episode
    from doomworm.experiments.worm_agent import WormScenario

    def motors(agent_factory: Callable[[], BrainLike]) -> list[tuple[float, float]]:
        world = build_world(3000, "doom4", "doom")
        return [r.motors for r in run_brain_episode(world, agent_factory(), 60)]

    def worm(with_route: bool) -> WormBrain:
        mapping = default_sensory_mapping()
        if not with_route:
            mapping.routes = [r for r in mapping.routes if not r.channel.startswith("novelty_")]
        scenario = WormScenario(sensory_mapping=mapping, maps="doom4", task="doom", sensors="ideal")
        return WormBrain.from_scenario(scenario)

    # with the route, the memory changes what the worm does
    assert motors(lambda: MemoryLayer(worm(True))) != motors(lambda: worm(True))
    # without it, the layer is invisible: it adds channels nobody reads
    assert motors(lambda: MemoryLayer(worm(False))) == pytest.approx(motors(lambda: worm(False)))

    # and an rnn, whose inputs were fixed when it was built, never sees them either
    rnn = build_candidate(CandidateSpec("rnn", maps="doom4", task="doom", sensors="ideal"))
    assert isinstance(rnn, RNNBrain)
    assert not any(name.startswith("novelty_") for name in rnn.inputs)
