"""Observation channels -> input currents for named neurons (Plan §2.2, §13)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

Target = tuple[str, float]  # (neuron_id, gain)


def _normalise(spec: Target | Sequence[Target]) -> list[Target]:
    """Accept a single ``(neuron, gain)`` or a sequence of them."""
    if len(spec) == 2 and isinstance(spec[0], str) and isinstance(spec[1], int | float):
        return [(spec[0], float(spec[1]))]
    targets: list[Target] = []
    for item in spec:
        if not isinstance(item, tuple):
            raise TypeError(f"expected (neuron, gain) pairs, got {item!r}")
        targets.append((str(item[0]), float(item[1])))
    return targets


class SensoryAdapter:
    """Linear, explicit, configurable mapping from channels to neurons.

    Args:
        channels: ``{channel_name: (neuron_id, gain)}`` or
            ``{channel_name: [(neuron_id, gain), ...]}``. Every listed neuron
            receives ``gain * observation[channel_name]``; a neuron named by
            several channels sums their contributions.
        tonic: Constant currents injected every tick regardless of the
            observation (e.g. a drive that keeps motor neurons active).
    """

    def __init__(
        self,
        channels: Mapping[str, Target | Sequence[Target]],
        tonic: Mapping[str, float] | None = None,
    ) -> None:
        self.routes: list[tuple[str, str, float]] = []
        for channel, spec in channels.items():
            for neuron_id, gain in _normalise(spec):
                self.routes.append((channel, neuron_id, gain))
        self.tonic = dict(tonic or {})

    @property
    def channels(self) -> dict[str, list[Target]]:
        """Routes grouped by channel."""
        out: dict[str, list[Target]] = {}
        for channel, neuron_id, gain in self.routes:
            out.setdefault(channel, []).append((neuron_id, gain))
        return out

    @property
    def neurons(self) -> set[str]:
        """Every neuron this adapter can drive."""
        return {n for _, n, _ in self.routes} | set(self.tonic)

    def __call__(self, observation: Mapping[str, float]) -> dict[str, float]:
        """Convert one observation into per-neuron input currents."""
        currents = dict(self.tonic)
        for channel, neuron_id, gain in self.routes:
            value = observation.get(channel, 0.0)
            currents[neuron_id] = currents.get(neuron_id, 0.0) + gain * value
        return currents
