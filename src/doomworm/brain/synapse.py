"""Weighted directed connection between two neurons."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Synapse:
    """Connection ``source -> target`` with a scalar weight.

    Weight is the only trainable quantity in the whole project (Plan §16).
    ``kind`` is ``"chemical"`` or ``"electrical"`` (Plan §11); the simulator
    treats both the same until gap junctions get their own model.
    """

    source: str
    target: str
    weight: float = 1.0
    kind: str = "chemical"

    def transmit(self, activity: float) -> float:
        """Current delivered to the target for a given source activity."""
        return self.weight * activity
