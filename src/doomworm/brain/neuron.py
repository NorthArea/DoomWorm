"""Discrete leaky integrate-and-fire neuron (Plan §2.3, §6)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Neuron:
    """A single neuron.

    Attributes:
        id: Unique name within a network.
        threshold: Potential at which the neuron fires. Must be > 0.
        decay: Fraction of potential lost per step, in [0, 1]. 0 = perfect
            integrator, 1 = no memory between steps.
        potential: Current membrane potential.
        activity: Output of the last :meth:`update` (1.0 fired, 0.0 silent).
    """

    id: str
    threshold: float = 1.0
    decay: float = 0.5
    potential: float = 0.0
    activity: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.decay <= 1.0:
            raise ValueError(f"decay must be in [0, 1], got {self.decay}")
        if self.threshold <= 0.0:
            raise ValueError(f"threshold must be > 0, got {self.threshold}")

    def integrate(self, current: float) -> None:
        """Leak the potential, then add the incoming current."""
        self.potential = self.potential * (1.0 - self.decay) + current

    def update(self) -> float:
        """Fire if the threshold is reached; return the new activity."""
        if self.potential >= self.threshold:
            self.activity = 1.0
            self.potential = 0.0
        else:
            self.activity = 0.0
        return self.activity
