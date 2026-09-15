"""The C. elegans connectome as a Brain: sensory mapping -> network -> motor mapping."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from doomworm.adapters import SensoryAdapter
from doomworm.brain import Network, Simulator, load_brain, save_brain
from doomworm.brains.base import Wheels
from doomworm.episode import MotorLike, average_activity


class WormBrain:
    """Runs ``brain_steps`` network ticks per environment step and reads the motor groups."""

    def __init__(
        self,
        network: Network,
        sensory: SensoryAdapter,
        motor: MotorLike,
        brain_steps: int = 5,
        name: str = "worm",
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.network = network
        self.sensory = sensory
        self.motor = motor
        self.brain_steps = brain_steps
        self.name = name
        self.meta = dict(meta or {})
        self.activity: dict[str, float] = {}
        self.sim = Simulator(network)

    def reset(self) -> None:
        """Clear neuron state and start a fresh simulator."""
        self.network.reset()
        self.sim = Simulator(self.network)
        self.activity = {}

    def act(self, channels: Mapping[str, float]) -> Wheels:
        """One environment step: same input for ``brain_steps`` ticks, mean activity -> wheels."""
        currents = self.sensory(channels)
        window = [self.sim.step(currents) for _ in range(self.brain_steps)]
        self.activity = average_activity(window)
        return self.motor(self.activity)

    # --- persistence -----------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Brain JSON (Plan §40) with the scenario params in meta."""
        save_brain(path, self.network, self.meta)

    @classmethod
    def from_scenario(cls, scenario: Any, name: str = "worm") -> WormBrain:
        """Wrap a :class:`~doomworm.experiments.worm_agent.WormScenario`."""
        return cls(
            scenario.template,
            scenario.sensory,
            scenario.motor,
            scenario.brain_steps,
            name=name,
            meta={"scenario": "worm", "params": scenario.params},
        )

    @classmethod
    def from_file(cls, path: Path | str, **overrides: Any) -> WormBrain:
        """Load a saved worm brain; ``overrides`` replace scenario params (maps, task, ...)."""
        from doomworm.experiments.worm_agent import WormScenario

        net, meta = load_brain(path)
        params = dict(meta.get("params", {})) | overrides
        scenario = WormScenario(**params)
        scenario.template.set_weights(net.get_weights())
        return cls(
            scenario.template,
            scenario.sensory,
            scenario.motor,
            scenario.brain_steps,
            name=Path(path).stem,
            meta={"scenario": "worm", "params": scenario.params} | {"source": str(path)},
        )
