"""Sensory and motor mappings between the world and C. elegans neurons (Plan §13-14).

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
    target_*        -> same neurons as food_* (Plan §19: the target is the
                       new attractive signal; the worm cannot tell them apart)
    danger_*        -> ASHL / ASHR / both  (nociceptive; used from stage 14)
    novelty_*       -> OLLL / OLLR / both  (stage 16: unvisited ground, from the memory
                                            layer -- the worm has no place memory of its own)
    prey_*          -> CEPDL / CEPDR / both (the enemy as an attractant, so the
                                            worm's own taxis can turn the body onto it)
    aim             -> ADLL, ADLR          (track B: enemy on the gun line)
                       RIPL, RIPR          (the only connection into the pharynx, where
                                            the trigger group lives: stage B2c)

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
TARGET = {"left": "target_left", "front": "target_front", "right": "target_right"}
PREY = {"left": "prey_left", "front": "prey_front", "right": "prey_right"}
PREY_GAIN = 3.0  # measured: the response saturates here (LIF activity clips at 1)
NOVELTY = {"left": "novelty_left", "front": "novelty_front", "right": "novelty_right"}
NOVELTY_GAIN = 3.0
DOCK = {"left": "dock_left", "front": "dock_front", "right": "dock_right"}


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

    def scale(self, channel_prefix: str, factor: float) -> SensoryMapping:
        """Multiply the gain of every route whose channel starts with ``channel_prefix``."""
        scaled = []
        for r in self.routes:
            if r.channel.startswith(channel_prefix):
                r = Route(r.channel, r.neuron, r.gain * factor)
            scaled.append(r)
        self.routes = scaled
        return self

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


@dataclass
class MotorMapping:
    """Neuron groups read by the motor adapter (Plan §14).

    ``forward`` and ``reversal`` set the drive, ``turn_left`` / ``turn_right``
    the differential. Dorsal (SMDD) is read as left and ventral (SMDV) as
    right: an admitted artifice, the worm bends its head dorso-ventrally.
    ``fire`` (Plan §22, track B) is the trigger group: pharyngeal pumping
    motor neurons, read as "bite"; empty for a brain without a gun.
    """

    forward: list[str] = field(default_factory=list)
    reversal: list[str] = field(default_factory=list)
    turn_left: list[str] = field(default_factory=list)
    turn_right: list[str] = field(default_factory=list)
    fire: list[str] = field(default_factory=list)

    def groups(self) -> dict[str, list[str]]:
        """Group name -> neurons."""
        return {
            "forward": self.forward,
            "reversal": self.reversal,
            "turn_left": self.turn_left,
            "turn_right": self.turn_right,
            "fire": self.fire,
        }

    def neurons(self) -> set[str]:
        """Every neuron referenced."""
        return {n for g in self.groups().values() for n in g}

    def validate(self, connectome: Connectome) -> None:
        """Raise if any neuron is missing or any drive group is empty."""
        missing = sorted(n for n in self.neurons() if n not in connectome)
        if missing:
            raise KeyError(f"motor mapping references unknown neurons: {missing}")
        empty = [name for name, group in self.groups().items() if not group and name != "fire"]
        if empty:
            raise ValueError(f"empty motor groups: {empty}")

    def describe(self) -> str:
        """Human-readable table."""
        return "\n".join(f"{name:<11} <- {', '.join(g)}" for name, g in self.groups().items())

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {name: list(g) for name, g in self.groups().items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MotorMapping:
        """Inverse of :meth:`to_dict`."""
        return cls(**{name: list(data.get(name, [])) for name in cls().groups()})

    def save(self, path: Path | str) -> None:
        """Write JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path | str) -> MotorMapping:
        """Read JSON."""
        return cls.from_dict(json.loads(Path(path).read_text()))


def default_motor_mapping() -> MotorMapping:
    """The Plan §14 table: command interneurons plus A/B motor classes, SMD/RIV for turns."""
    return MotorMapping(
        forward=["AVBL", "AVBR", "PVCL", "PVCR"]
        + [f"VB{i:02d}" for i in range(1, 12)]
        + [f"DB{i:02d}" for i in range(1, 8)],
        reversal=["AVAL", "AVAR", "AVDL", "AVDR"]
        + [f"VA{i:02d}" for i in range(1, 13)]
        + [f"DA{i:02d}" for i in range(1, 10)],
        turn_left=["SMDDL", "SMDDR", "RIVL"],
        turn_right=["SMDVL", "SMDVR", "RIVR"],
        fire=["M3L", "M3R", "M4", "MCL", "MCR"],
    )


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
    m.add(TARGET["left"], ["AWAL", "AWCL", "ASEL"])
    m.add(TARGET["right"], ["AWAR", "AWCR", "ASER"])
    m.add(TARGET["front"], ["AWAL", "AWCL", "ASEL", "AWAR", "AWCR", "ASER"])
    m.add(DOCK["left"], ["AWAL", "AWCL", "ASEL"])
    m.add(DOCK["right"], ["AWAR", "AWCR", "ASER"])
    m.add(DOCK["front"], ["AWAL", "AWCL", "ASEL", "AWAR", "AWCR", "ASER"])
    m.add(DANGER["left"], ["ASHL"])
    m.add(DANGER["right"], ["ASHR"])
    m.add(DANGER["front"], ["ASHL", "ASHR"])
    # Track B: the gun line. The amphid pair reads it, and the same signal is delivered to
    # RIPL/RIPR because they are the animal's only door into the pharynx (5 crossing
    # connections in the whole connectome), where the trigger group M3/M4/MC lives. With
    # ADL alone the trigger is blind to the target: untrained fire 0.134 with an enemy on
    # the line against 0.157 with none. Through RIP the same untrained worm answers 0.906
    # against 0.143 (docs/assumptions.md 2026-09-16, stage B2c).
    m.add("aim", ["ADLL", "ADLR", "RIPL", "RIPR"])
    # The same monster that ASH reads as pain, read again as prey, by side. A body with
    # the gun bolted to it can only aim by turning, and turning toward something is what
    # the chemosensory pathway does; the escape pathway does the opposite.
    #
    # The pair was chosen by outcome, not by the story. Scoring the 31 free sensory pairs
    # by hits landed on a standing target from five bearings (200 ticks each, untrained):
    # CEPD 20 hits / 27 shots, PLM 12/22, ADE 4/20, everything else <= 3, and the bare
    # connectome 0. Adding any second pair to CEPD destroys it (0 hits). CEPD is also the
    # cleanest orienting pair measured on its own (+0.037 / -0.023 mean wheel difference
    # with the enemy 50 degrees off the bow, left case / right case).
    #
    # The biology agrees: CEP are the dopaminergic head mechanosensors that report a
    # bacterial lawn under the nose and slow the animal on food. "Prey right here, slow
    # down and turn onto it" is the behaviour they already own. Gain 3; 6 changes nothing
    # (graded activity clips at 1). Stage B2d.
    m.add(PREY["left"], ["CEPDL"], PREY_GAIN)
    m.add(PREY["right"], ["CEPDR"], PREY_GAIN)
    m.add(PREY["front"], ["CEPDL", "CEPDR"], PREY_GAIN)
    # Stage 16: where the agent has *not* been, supplied by the memory layer as a
    # gradient in the same form as the food smell. The animal has no place memory;
    # the layer holds it and the brain only reads it. OLL was the best of the free
    # sensory pairs by reward on an untrained worm (doom1 -18.5 -> +0.5, doom2
    # -19.0 -> -2.5, doom4 -21.1 -> -5.9); a brain with no route here is unchanged.
    # Stage 20: the *change* in a gradient, which is what the animal actually senses --
    # ASEL fires when salt is rising, ASER when it is falling, and the falling edge is
    # what raises the turn rate in a pirouette. Every other channel here is a level.
    m.add("target_rising", ["ASEL"])
    m.add("target_falling", ["ASER"])
    m.add("prey_rising", ["ASEL"])
    m.add("prey_falling", ["ASER"])
    m.add(NOVELTY["left"], ["OLLL"], NOVELTY_GAIN)
    m.add(NOVELTY["right"], ["OLLR"], NOVELTY_GAIN)
    m.add(NOVELTY["front"], ["OLLL", "OLLR"], NOVELTY_GAIN)
    return m


def interface_synapses(connectome: Connectome, scope: str = "interface") -> list[int]:
    """Indices of the synapses a restricted search may move (stage 18).

    ``sensory``    the synapses leaving the neurons a channel is injected into (506)
    ``interface``  those plus the synapses entering a motor group (1819)
    ``all``        every synapse (5905)

    The connectome is too well connected for reachability to narrow anything --
    within two hops of the sensors, 99.3 % of synapses can already influence a
    motor group -- so the useful cut is the interface: how loudly the senses
    speak and how loudly the muscles listen, with the animal's interneurons left
    exactly as they are.
    """
    if scope not in ("sensory", "interface", "all"):
        raise ValueError("scope must be 'sensory', 'interface' or 'all'")
    if scope == "all":
        return list(range(len(connectome.connections)))
    sensory = default_sensory_mapping()
    sources = {n for channel in sensory.channels() for n, _ in sensory.targets(channel)}
    motor = default_motor_mapping()
    sinks = set(motor.forward) | set(motor.reversal) | set(motor.turn_left)
    sinks |= set(motor.turn_right) | set(motor.fire)
    chosen = []
    for index, connection in enumerate(connectome.connections):
        if connection.source in sources or (scope == "interface" and connection.target in sinks):
            chosen.append(index)
    return chosen
