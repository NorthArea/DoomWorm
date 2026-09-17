"""Synchronous discrete-time simulation of a :class:`Network`.

The dynamics are those of :class:`~broomworm.brain.neuron.Neuron` applied to
every neuron at once: currents come from the *previous* tick's activities,
so a signal travels one synapse per tick and update order is irrelevant.
State is kept in NumPy arrays for speed and written back to the neuron
objects after every tick, so ``Network.activities()`` and the neurons'
``potential`` / ``activity`` fields stay valid.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from broomworm.brain.network import Network


class Simulator:
    """Advance a network one tick at a time."""

    def __init__(self, network: Network) -> None:
        self.network = network
        self.time = 0
        self.ids = list(network.neurons)
        self._index = {nid: i for i, nid in enumerate(self.ids)}
        n = len(self.ids)
        neurons = list(network.neurons.values())
        self._threshold = np.array([nr.threshold for nr in neurons])
        self._keep = np.array([1.0 - nr.decay for nr in neurons])
        self._graded = np.array([nr.graded for nr in neurons], dtype=bool)
        self._potential = np.array([nr.potential for nr in neurons])
        self._activity = np.array([nr.activity for nr in neurons])
        # weights[target, source]: multiple synapses between a pair add up
        self._weights = np.zeros((n, n))
        for s in network.synapses:
            self._weights[self._index[s.target], self._index[s.source]] += s.weight
        self._input = np.zeros(n)

    def step(self, inputs: Mapping[str, float] | None = None) -> dict[str, float]:
        """Apply ``inputs`` (external current per neuron id) and advance one tick."""
        self._input.fill(0.0)
        for nid, value in (inputs or {}).items():
            if nid not in self._index:
                raise KeyError(f"unknown neuron {nid!r}")
            self._input[self._index[nid]] += value

        current = self._input + self._weights @ self._activity
        self._potential = self._potential * self._keep + current

        graded = np.clip(self._potential / self._threshold, 0.0, 1.0)
        fired = self._potential >= self._threshold
        self._activity = np.where(self._graded, graded, fired.astype(float))
        self._potential = np.where(~self._graded & fired, 0.0, self._potential)

        for nid, p, a in zip(self.ids, self._potential, self._activity, strict=True):
            neuron = self.network.neurons[nid]
            neuron.potential = float(p)
            neuron.activity = float(a)
        self.time += 1
        return dict(zip(self.ids, self._activity.tolist(), strict=True))

    def run(self, steps: int, inputs: Mapping[str, float] | None = None) -> list[dict[str, float]]:
        """Step ``steps`` times with constant ``inputs``; return activity per tick."""
        return [self.step(inputs) for _ in range(steps)]
