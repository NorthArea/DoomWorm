"""Synchronous discrete-time simulation of a :class:`Network`."""

from __future__ import annotations

from collections.abc import Mapping

from doomworm.brain.network import Network


class Simulator:
    """Advance a network one tick at a time.

    Each step uses the activities from the *previous* step, so a signal
    travels exactly one synapse per tick and the update order of neurons
    does not matter.
    """

    def __init__(self, network: Network) -> None:
        self.network = network
        self.time = 0

    def step(self, inputs: Mapping[str, float] | None = None) -> dict[str, float]:
        """Apply ``inputs`` (external current per neuron id) and advance one tick."""
        inputs = dict(inputs or {})
        for nid in inputs:
            if nid not in self.network.neurons:
                raise KeyError(f"unknown neuron {nid!r}")

        previous = self.network.activities()
        for nid, neuron in self.network.neurons.items():
            current = inputs.get(nid, 0.0)
            for syn in self.network.incoming(nid):
                current += syn.transmit(previous[syn.source])
            neuron.integrate(current)

        for neuron in self.network.neurons.values():
            neuron.update()

        self.time += 1
        return self.network.activities()

    def run(self, steps: int, inputs: Mapping[str, float] | None = None) -> list[dict[str, float]]:
        """Step ``steps`` times with constant ``inputs``; return activity per tick."""
        return [self.step(inputs) for _ in range(steps)]
