"""Stage 23: teaching the brain what the search found."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from wormlab.body import Drive
from wormlab.candidates import ScriptedBrain
from wormlab.learning.distill import (
    Demonstration,
    imitation_error,
    imitation_fitness,
    load_demos,
    save_demos,
)


def demo(actions: list[Drive]) -> Demonstration:
    d = Demonstration()
    for i, action in enumerate(actions):
        d.add({"sensor_front": 0.1 * i}, action)
    return d


def test_a_brain_that_already_agrees_has_no_error() -> None:
    target = Drive(forward=0.5, turn=0.25)
    brain = ScriptedBrain(lambda _c: (0.25, 0.75), "exact")  # forward 0.5, turn 0.25
    assert imitation_error(brain, [demo([target] * 3)]) == pytest.approx(0.0)
    assert imitation_fitness(brain, [demo([target] * 3)]) == pytest.approx(0.0)


def test_the_error_is_the_distance_in_intent() -> None:
    brain = ScriptedBrain(lambda _c: (0.0, 0.0), "still")
    error = imitation_error(brain, [demo([Drive(forward=1.0, turn=0.5)] * 4)])
    assert error == pytest.approx(1.5), "one unit of forward plus half a unit of turn"


def test_fitness_prefers_the_closer_brain() -> None:
    demos = [demo([Drive(forward=1.0)] * 5)]
    close = ScriptedBrain(lambda _c: (0.9, 0.9), "close")
    far = ScriptedBrain(lambda _c: (-1.0, -1.0), "far")
    assert imitation_fitness(close, demos) > imitation_fitness(far, demos)


def test_demonstrations_round_trip_through_a_file(tmp_path: Path) -> None:
    demos = [demo([Drive(forward=0.2, turn=-0.3, fire=0.9)] * 2), demo([Drive()] * 3)]
    path = save_demos(demos, tmp_path / "demos.json")
    loaded = load_demos(path)
    assert [len(d) for d in loaded] == [2, 3]
    assert loaded[0].actions[0].fire == pytest.approx(0.9)
    assert loaded[0].channels[1]["sensor_front"] == pytest.approx(0.1)


def test_each_episode_starts_the_brain_afresh() -> None:
    """A recurrent brain fed a trajectory out of order is answering nothing."""
    seen: list[int] = []

    class Counter:
        name = "counter"

        def __init__(self) -> None:
            self.ticks = 0

        def reset(self) -> None:
            seen.append(self.ticks)
            self.ticks = 0

        def act(self, _channels: Mapping[str, float]) -> tuple[float, float]:
            self.ticks += 1
            return 0.0, 0.0

    imitation_error(Counter(), [demo([Drive()] * 3), demo([Drive()] * 5)])
    assert seen == [0, 3], "reset before each episode, after three ticks of the first"
