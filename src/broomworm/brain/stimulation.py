"""Stimulate named neurons and record how activity spreads (Plan §12)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from broomworm.brain.network import Network
from broomworm.brain.simulator import Simulator


@dataclass
class ActivityTrace:
    """Per-tick activity of every neuron during and after a stimulus."""

    history: list[dict[str, float]]
    stimulus: dict[str, float]
    hold: int

    @property
    def ticks(self) -> int:
        """Number of recorded ticks."""
        return len(self.history)

    def peak(self) -> dict[str, float]:
        """Maximum activity per neuron over the trace."""
        peaks: dict[str, float] = dict.fromkeys(self.history[0], 0.0)
        for step in self.history:
            for nid, value in step.items():
                if value > peaks[nid]:
                    peaks[nid] = value
        return peaks

    def first_active(self, threshold: float = 0.01) -> dict[str, int]:
        """First tick at which each neuron exceeded ``threshold`` (absent if never)."""
        first: dict[str, int] = {}
        for tick, step in enumerate(self.history):
            for nid, value in step.items():
                if value > threshold and nid not in first:
                    first[nid] = tick
        return first

    def top(self, k: int = 10, exclude_stimulus: bool = True) -> list[tuple[str, float]]:
        """The ``k`` neurons with the highest peak activity."""
        peaks = self.peak()
        items = [(n, v) for n, v in peaks.items() if not (exclude_stimulus and n in self.stimulus)]
        return sorted(items, key=lambda kv: -kv[1])[:k]

    def max_activity(self, tick: int) -> float:
        """Largest activity of any neuron at ``tick``."""
        return max(self.history[tick].values())


def stimulate(
    net: Network,
    stimulus: Mapping[str, float],
    hold: int = 10,
    settle: int = 20,
    reset: bool = True,
) -> ActivityTrace:
    """Inject ``stimulus`` for ``hold`` ticks, then run ``settle`` ticks with no input."""
    if reset:
        net.reset()
    sim = Simulator(net)
    history = [sim.step(stimulus) for _ in range(hold)]
    history += [sim.step() for _ in range(settle)]
    return ActivityTrace(history=history, stimulus=dict(stimulus), hold=hold)
