"""Stage 23: teach the brain what the search found.

Stage 21 gave us a teacher that is reliably better than the student — a rollout
over the connectome's own proposals — and stage 22 said where it overrules the
student most. This is the step that uses that: record what the teacher chose,
then move the weights until the brain chooses it by itself.

There is no gradient through a leaky-integrate-and-fire network here, so the
"training" is the search we already have with a different fitness. Instead of
reward it scores **how closely the brain reproduces the teacher's action on the
teacher's own states**, which turns an expensive look-ahead into something the
brain can do alone, in one tick, on a robot.

The demonstrations keep their episode order and the brain is reset at the start
of each one, because a recurrent network fed shuffled states is answering a
question nobody asked.

What this settles: every negative result in this project has been ambiguous
between *the architecture cannot represent this behaviour* and *our search did
not find it*. If the brain can be taught what the rollout found, the first
reading is dead.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wormlab.body import Drive, drive_of
from wormlab.episode import BrainLike


@dataclass
class Demonstration:
    """One episode of the teacher: states in order, and what it did in each."""

    channels: list[dict[str, float]] = field(default_factory=list)
    actions: list[Drive] = field(default_factory=list)

    def add(self, channels: dict[str, float], action: Drive) -> None:
        """Append one tick."""
        self.channels.append(dict(channels))
        self.actions.append(action)

    def __len__(self) -> int:
        return len(self.actions)

    def to_dict(self) -> dict[str, object]:
        """JSON form: the states, and the actions as four numbers each."""
        return {
            "channels": self.channels,
            "actions": [[a.forward, a.turn, a.strafe, a.fire] for a in self.actions],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Demonstration:
        """Inverse of :meth:`to_dict`."""
        actions = [Drive(*values) for values in data["actions"]]
        return cls(channels=[dict(c) for c in data["channels"]], actions=actions)


def save_demos(demos: Sequence[Demonstration], path: Path | str) -> Path:
    """Write a teacher's demonstrations where training can pick them up."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([d.to_dict() for d in demos]) + "\n")
    return out


def load_demos(path: Path | str) -> list[Demonstration]:
    """Read them back."""
    return [Demonstration.from_dict(d) for d in json.loads(Path(path).read_text())]


def imitation_error(brain: BrainLike, demos: Sequence[Demonstration]) -> float:
    """Mean distance between what the brain does and what the teacher did.

    The episode order is kept and the brain is reset per episode: a recurrent
    network only means anything along a trajectory.
    """
    total = 0.0
    ticks = 0
    for demo in demos:
        brain.reset()
        for channels, target in zip(demo.channels, demo.actions, strict=True):
            intent = drive_of(brain.act(channels), float(getattr(brain, "fire", 0.0)))
            total += abs(intent.forward - target.forward) + abs(intent.turn - target.turn)
            ticks += 1
    return total / max(1, ticks)


def imitation_fitness(brain: BrainLike, demos: Sequence[Demonstration]) -> float:
    """The error as something to maximise, so the existing search can drive it."""
    return -imitation_error(brain, demos)
