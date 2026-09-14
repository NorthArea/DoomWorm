"""Observation channels -> input currents for named neurons (Plan §2.2)."""

from __future__ import annotations

from collections.abc import Mapping


class SensoryAdapter:
    """Linear, explicit, configurable mapping from channels to neurons.

    Args:
        channels: ``{channel_name: (neuron_id, gain)}``. The current injected
            into ``neuron_id`` is ``gain * observation[channel_name]``.
        tonic: Constant currents injected every tick regardless of the
            observation (e.g. a drive that keeps motor neurons active).
    """

    def __init__(
        self,
        channels: Mapping[str, tuple[str, float]],
        tonic: Mapping[str, float] | None = None,
    ) -> None:
        self.channels = dict(channels)
        self.tonic = dict(tonic or {})

    def __call__(self, observation: Mapping[str, float]) -> dict[str, float]:
        """Convert one observation into per-neuron input currents."""
        currents = dict(self.tonic)
        for channel, (neuron_id, gain) in self.channels.items():
            currents[neuron_id] = currents.get(neuron_id, 0.0) + gain * observation.get(
                channel, 0.0
            )
        return currents
