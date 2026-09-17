"""The C. elegans connectome as a Brain: sensory mapping -> network -> motor mapping."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from doomworm.adapters import FireAdapter, SensoryAdapter
from doomworm.brain import Network, Simulator, load_brain, save_brain
from doomworm.candidates.base import Wheels
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
        trigger: FireAdapter | None = None,
        trainable: Sequence[int] | None = None,
        tune_gains: bool = False,
    ) -> None:
        self.network = network
        self.sensory = sensory
        self.motor = motor
        self.trigger = trigger
        self.brain_steps = brain_steps
        self.name = name
        self.meta = dict(meta or {})
        self.activity: dict[str, float] = {}
        self.fire = 0.0
        # Which synapses the search may move. None = all of them; a subset keeps the
        # animal's interneurons as they are and tunes only the interface (stage 18).
        self.trainable = None if trainable is None else list(trainable)
        # Stage 18: the adapter's gains as genes. They are translation, not network --
        # how loudly a motor group is heard and how easily the trigger goes -- and
        # the untrained probe showed the turn gain decides whether the body comes
        # round at all. A gene in [-1, 1] becomes `base * 8 ** gene`, so the shipped
        # value sits at 0 and the search may reach an eighth of it or eight times it.
        self.tune_gains = tune_gains
        self.gain_base = (
            float(getattr(motor, "gain_drive", 10.0)),
            float(getattr(motor, "gain_turn", 10.0)),
            float(getattr(trigger, "gain", 10.0)),
        )
        self.gain_genes = [0.0, 0.0, 0.0]
        self.sim = Simulator(network)

    def reset(self) -> None:
        """Clear neuron state and start a fresh simulator."""
        self.network.reset()
        self.sim = Simulator(self.network)
        self.activity = {}
        self.fire = 0.0

    def act(self, channels: Mapping[str, float]) -> Wheels:
        """One environment step: same input for ``brain_steps`` ticks, mean activity -> wheels."""
        currents = self.sensory(channels)
        window = [self.sim.step(currents) for _ in range(self.brain_steps)]
        self.activity = average_activity(window)
        if self.trigger is not None:
            self.fire = self.trigger(self.activity)
        return self.motor(self.activity)

    # --- Trainable ----------------------------------------------------------------

    @property
    def n_synapse_genes(self) -> int:
        """Synapses the search may move."""
        return len(self.network.synapses) if self.trainable is None else len(self.trainable)

    @property
    def n_weights(self) -> int:
        """The genome: trainable synapses, plus the adapter gains when they are tuned."""
        return self.n_synapse_genes + (len(self.gain_genes) if self.tune_gains else 0)

    def get_weights(self) -> list[float]:
        """The genome, synapses first."""
        weights = self.network.get_weights()
        genes = weights if self.trainable is None else [weights[i] for i in self.trainable]
        return [*genes, *self.gain_genes] if self.tune_gains else genes

    def set_weights(self, weights: Sequence[float]) -> None:
        """Load a genome; untrainable synapses keep the value the animal has."""
        values = [float(w) for w in weights]
        if self.tune_gains:
            values, self.gain_genes = values[: self.n_synapse_genes], values[self.n_synapse_genes :]
            self._apply_gains()
        if self.trainable is None:
            self.network.set_weights(values)
        else:
            current = self.network.get_weights()
            for slot, value in zip(self.trainable, values, strict=True):
                current[slot] = value
            self.network.set_weights(current)
        self.sim = Simulator(self.network)

    def _apply_gains(self) -> None:
        """Genes in [-1, 1] -> gains in [base / 8, base * 8]."""
        drive, turn, fire = (
            base * 8.0**gene for base, gene in zip(self.gain_base, self.gain_genes, strict=True)
        )
        self.motor.gain_drive = drive  # type: ignore[attr-defined]
        self.motor.gain_turn = turn  # type: ignore[attr-defined]
        if self.trigger is not None:
            self.trigger.gain = fire

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
            trigger=scenario.trigger,
        )

    @classmethod
    def from_file(cls, path: Path | str, **overrides: Any) -> WormBrain:
        """Load a saved worm brain; ``overrides`` replace scenario params (maps, task, ...)."""
        from doomworm.experiments.worm_agent import WormScenario

        net, meta = load_brain(path)
        params = dict(meta.get("params", {})) | overrides
        scenario = WormScenario(**params)
        # The saved network is used as is (a control topology differs from the
        # connectome); the scenario only contributes the named-neuron adapters.
        keep = {k: meta[k] for k in ("variant", "candidate", "layer", "train") if k in meta}
        return cls(
            net,
            scenario.sensory,
            scenario.motor,
            scenario.brain_steps,
            name=Path(path).stem,
            meta={"scenario": "worm", "params": scenario.params, "source": str(path)} | keep,
            trigger=scenario.trigger,
        )
