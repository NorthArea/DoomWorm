"""Candidate brains of the A2 bake-off, built from a plain spec (Plan §20.4).

Every candidate is a :class:`~doomworm.brains.base.Brain` that is also
:class:`~doomworm.brains.base.Trainable`; the training harness only sees the
weight vector. Adding a candidate means adding a name here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from doomworm.brains.rnn import RNNBrain
from doomworm.brains.worm import WormBrain

WORM_VARIANTS = ("worm", "worm_random", "worm_shuffled", "worm_dense")
CANDIDATES = (*WORM_VARIANTS, "rnn")
Candidate = WormBrain | RNNBrain


@dataclass(frozen=True)
class CandidateSpec:
    """What to build: a candidate name, optional saved weights, world parameters."""

    kind: str
    init: str | None = None  # brain JSON to start from (curriculum)
    variant_seed: int = 0  # seed of a control topology
    maps: str = "apartment"
    task: str = "clean"
    sensors: str = "vacuum"
    dangers: int = 0
    params: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def label(self) -> str:
        """Row name for the leaderboard."""
        return self.kind if self.init is None else f"{self.kind}_from_{Path(self.init).stem}"


def load_candidate(path: Path | str, **world: Any) -> Candidate:
    """Load any saved candidate brain by its file format."""
    text = Path(path).read_text(encoding="utf-8")
    if '"kind": "rnn"' in text[:200]:
        return RNNBrain.from_file(path)
    return WormBrain.from_file(path, **world)


def build_candidate(spec: CandidateSpec) -> Candidate:
    """Instantiate a candidate; the saved brain (if any) supplies the weights."""
    if spec.kind not in CANDIDATES:
        raise ValueError(f"unknown candidate {spec.kind!r}, choose from {CANDIDATES}")
    if spec.init is not None:
        brain = load_candidate(
            spec.init, maps=spec.maps, task=spec.task, sensors=spec.sensors, dangers=spec.dangers
        )
        brain.meta["candidate"] = spec.kind
        return brain
    if spec.kind == "rnn":
        from doomworm.environments.sensors import PRESETS, SensorSuite

        inputs = SensorSuite(PRESETS[spec.sensors]).channel_names
        return RNNBrain(inputs, seed=spec.variant_seed, **spec.params)
    from doomworm.experiments.worm_agent import WormScenario

    connectome = None
    if spec.kind != "worm":
        from doomworm.connectome import load_cook2019
        from doomworm.connectome.variants import make_variant

        connectome = make_variant(
            load_cook2019(), spec.kind.removeprefix("worm_"), spec.variant_seed
        )
    scenario = WormScenario(
        connectome=connectome,
        maps=spec.maps,
        task=spec.task,
        sensors=spec.sensors,
        dangers=spec.dangers,
        **spec.params,
    )
    brain = WormBrain.from_scenario(scenario, name=spec.kind)
    brain.meta["candidate"] = spec.kind
    if spec.kind != "worm":
        brain.meta["variant"] = {"kind": spec.kind, "seed": spec.variant_seed}
    return brain
