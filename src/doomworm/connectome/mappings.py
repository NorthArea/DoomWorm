"""Sensory mapping: world channels -> C. elegans sensory neurons (Plan §13).

The default table follows the plan. Left/right is an admitted artifice: the
worm has no lateralised touch or smell in this sense, it steers with dorsal /
ventral head bends. ``front`` drives both members of every pair.

    obstacle_left   -> ALML, FLPL          (mechanosensory, anterior touch)
    obstacle_right  -> ALMR, FLPR
    obstacle_front  -> AVM, FLPL, FLPR
    food_left       -> AWAL, AWCL, ASEL    (chemosensory, amphid)
    food_right      -> AWAR, AWCR, ASER
    food_front      -> all six
    hunger          -> NSML, NSMR, ASIL, ASIR
    danger_*        -> ASHL / ASHR / both  (nociceptive; used from stage 14)

Mappings serialise to JSON so an experiment can ship its own table.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from doomworm.adapters import SensoryAdapter
from doomworm.connectome.model import Connectome

# World observation channels (see environments.simple_2d.Observation.as_channels).
OBSTACLE = {"left": "sensor_left", "front": "sensor_front", "right": "sensor_right"}
FOOD = {"left": "food_left", "front": "food_front", "right": "food_right"}
DANGER = {"left": "danger_left", "front": "danger_front", "right": "danger_right"}


@dataclass(frozen=True)
class Route:
    """One channel driving one neuron with a gain."""

    channel: str
    neuron: str
    gain: float = 1.0


@dataclass
class SensoryMapping:
    """Ordered, explicit list of routes plus optional tonic currents."""

    routes: list[Route] = field(default_factory=list)
    tonic: dict[str, float] = field(default_factory=dict)

    def add(self, channel: str, neurons: Iterable[str], gain: float = 1.0) -> SensoryMapping:
        """Route ``channel`` to every neuron in ``neurons``."""
        for n in neurons:
            self.routes.append(Route(channel, n, gain))
        return self

    def neurons(self) -> set[str]:
        """Every neuron referenced."""
        return {r.neuron for r in self.routes} | set(self.tonic)

    def channels(self) -> list[str]:
        """Channels in first-seen order."""
        return list(dict.fromkeys(r.channel for r in self.routes))

    def targets(self, channel: str) -> list[tuple[str, float]]:
        """Neurons and gains driven by ``channel``."""
        return [(r.neuron, r.gain) for r in self.routes if r.channel == channel]

    def validate(self, connectome: Connectome) -> None:
        """Raise if any neuron is missing from the connectome."""
        missing = sorted(n for n in self.neurons() if n not in connectome)
        if missing:
            raise KeyError(f"mapping references unknown neurons: {missing}")

    def to_adapter(self) -> SensoryAdapter:
        """Build the runtime adapter."""
        channels: dict[str, list[tuple[str, float]]] = {}
        for r in self.routes:
            channels.setdefault(r.channel, []).append((r.neuron, r.gain))
        return SensoryAdapter(channels, tonic=self.tonic)

    def describe(self) -> str:
        """Human-readable table."""
        width = max((len(c) for c in self.channels()), default=8)
        lines = []
        for channel in self.channels():
            parts = ", ".join(
                f"{n}" if g == 1.0 else f"{n}x{g:g}" for n, g in self.targets(channel)
            )
            lines.append(f"{channel:<{width}}  -> {parts}")
        for neuron, current in self.tonic.items():
            lines.append(f"{'tonic':<{width}}  -> {neuron} {current:+g}")
        return "\n".join(lines)

    # --- persistence ---------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "routes": [
                {"channel": r.channel, "neuron": r.neuron, "gain": r.gain} for r in self.routes
            ],
            "tonic": dict(self.tonic),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SensoryMapping:
        """Inverse of :meth:`to_dict`."""
        routes = [
            Route(r["channel"], r["neuron"], float(r.get("gain", 1.0))) for r in data["routes"]
        ]
        return cls(routes=routes, tonic={k: float(v) for k, v in data.get("tonic", {}).items()})

    def save(self, path: Path | str) -> None:
        """Write JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path | str) -> SensoryMapping:
        """Read JSON."""
        return cls.from_dict(json.loads(Path(path).read_text()))


def default_sensory_mapping() -> SensoryMapping:
    """The Plan §13 table."""
    m = SensoryMapping()
    m.add(OBSTACLE["left"], ["ALML", "FLPL"])
    m.add(OBSTACLE["right"], ["ALMR", "FLPR"])
    m.add(OBSTACLE["front"], ["AVM", "FLPL", "FLPR"])
    m.add(FOOD["left"], ["AWAL", "AWCL", "ASEL"])
    m.add(FOOD["right"], ["AWAR", "AWCR", "ASER"])
    m.add(FOOD["front"], ["AWAL", "AWCL", "ASEL", "AWAR", "AWCR", "ASER"])
    m.add("hunger", ["NSML", "NSMR", "ASIL", "ASIR"])
    m.add(DANGER["left"], ["ASHL"])
    m.add(DANGER["right"], ["ASHR"])
    m.add(DANGER["front"], ["ASHL", "ASHR"])
    return m
