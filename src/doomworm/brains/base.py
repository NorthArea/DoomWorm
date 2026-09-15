"""The one interface every candidate brain implements (Plan §3.3)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, runtime_checkable

Wheels = tuple[float, float]


@runtime_checkable
class Brain(Protocol):
    """Observation channels in, wheel commands out. Stateful between ticks, reset per episode."""

    name: str

    def reset(self) -> None:
        """Forget episode state."""
        ...

    def act(self, channels: Mapping[str, float]) -> Wheels:
        """Return ``(wheel_left, wheel_right)`` in [-1, 1] for this environment step."""
        ...


@runtime_checkable
class Trainable(Protocol):
    """A Brain whose behaviour is a flat weight vector (evolution trains it in place)."""

    @property
    def n_weights(self) -> int:
        """Genome length."""
        ...

    def get_weights(self) -> list[float]:
        """Current genome."""
        ...

    def set_weights(self, weights: Sequence[float]) -> None:
        """Load a genome."""
        ...


class ScriptedBrain:
    """Wrap a plain function as a Brain (tests, baselines, gradient followers)."""

    def __init__(self, fn: Callable[[Mapping[str, float]], Wheels], name: str = "scripted") -> None:
        self.fn = fn
        self.name = name

    def reset(self) -> None:
        """Stateless."""

    def act(self, channels: Mapping[str, float]) -> Wheels:
        """Delegate."""
        return self.fn(channels)
