"""Container for neurons and synapses. No dynamics live here."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from broomworm.brain.neuron import Neuron
from broomworm.brain.synapse import Synapse


class Network:
    """Directed graph of :class:`Neuron` nodes and :class:`Synapse` edges."""

    def __init__(self) -> None:
        self.neurons: dict[str, Neuron] = {}
        self.synapses: list[Synapse] = []
        self._incoming: defaultdict[str, list[Synapse]] = defaultdict(list)

    def add_neuron(self, neuron: Neuron) -> None:
        """Register a neuron; ids must be unique."""
        if neuron.id in self.neurons:
            raise ValueError(f"neuron {neuron.id!r} already exists")
        self.neurons[neuron.id] = neuron

    def add_synapse(self, synapse: Synapse) -> None:
        """Register a synapse; both endpoints must already exist."""
        for endpoint in (synapse.source, synapse.target):
            if endpoint not in self.neurons:
                raise KeyError(f"unknown neuron {endpoint!r}")
        self.synapses.append(synapse)
        self._incoming[synapse.target].append(synapse)

    def incoming(self, neuron_id: str) -> list[Synapse]:
        """Synapses whose target is ``neuron_id``."""
        return list(self._incoming.get(neuron_id, ()))

    def activities(self) -> dict[str, float]:
        """Snapshot of every neuron's current activity."""
        return {nid: n.activity for nid, n in self.neurons.items()}

    # --- trainable parameters ----------------------------------------------

    def get_weights(self) -> list[float]:
        """Synaptic weights in insertion order: the genome of this topology."""
        return [s.weight for s in self.synapses]

    def set_weights(self, weights: Sequence[float]) -> None:
        """Overwrite synaptic weights in insertion order."""
        if len(weights) != len(self.synapses):
            raise ValueError(f"expected {len(self.synapses)} weights, got {len(weights)}")
        for synapse, w in zip(self.synapses, weights, strict=True):
            synapse.weight = float(w)

    def reset(self) -> None:
        """Clear transient state (potential, activity) before a new episode."""
        for neuron in self.neurons.values():
            neuron.potential = 0.0
            neuron.activity = 0.0
