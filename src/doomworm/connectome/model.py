"""Internal connectome format (Plan §11). Nothing here depends on a dataset layout."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum


class ConnectionType(Enum):
    """Synapse class."""

    CHEMICAL = "chemical"
    ELECTRICAL = "electrical"


@dataclass(frozen=True)
class NeuronInfo:
    """One biological neuron."""

    id: str
    name: str
    type: str  # sensory | interneuron | motorneuron | neuron (unclassified)
    metadata: dict[str, str] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class Connection:
    """Directed connection with a raw dataset weight (strength, not a trained weight)."""

    source: str
    target: str
    weight: float
    connection_type: ConnectionType


@dataclass
class Connectome:
    """Neurons plus connections. Electrical connections are stored once per direction."""

    neurons: list[NeuronInfo]
    connections: list[Connection]

    def __post_init__(self) -> None:
        self._by_id = {n.id: n for n in self.neurons}
        if len(self._by_id) != len(self.neurons):
            raise ValueError("duplicate neuron ids")
        self._out: defaultdict[str, list[Connection]] = defaultdict(list)
        self._in: defaultdict[str, list[Connection]] = defaultdict(list)
        for c in self.connections:
            for endpoint in (c.source, c.target):
                if endpoint not in self._by_id:
                    raise KeyError(f"connection references unknown neuron {endpoint!r}")
            self._out[c.source].append(c)
            self._in[c.target].append(c)

    def __len__(self) -> int:
        return len(self.neurons)

    def __contains__(self, neuron_id: str) -> bool:
        return neuron_id in self._by_id

    def neuron(self, neuron_id: str) -> NeuronInfo:
        """Look up a neuron by id."""
        return self._by_id[neuron_id]

    def ids(self) -> list[str]:
        """All neuron ids in stored order."""
        return [n.id for n in self.neurons]

    def by_type(self, neuron_type: str) -> list[NeuronInfo]:
        """Neurons of one type."""
        return [n for n in self.neurons if n.type == neuron_type]

    def outgoing(self, neuron_id: str) -> list[Connection]:
        """Connections leaving ``neuron_id``."""
        return list(self._out.get(neuron_id, ()))

    def incoming(self, neuron_id: str) -> list[Connection]:
        """Connections entering ``neuron_id``."""
        return list(self._in.get(neuron_id, ()))

    def of_type(self, connection_type: ConnectionType) -> list[Connection]:
        """Connections of one class."""
        return [c for c in self.connections if c.connection_type is connection_type]

    def downstream(self, neuron_ids: Iterable[str]) -> set[str]:
        """Direct targets of a set of neurons."""
        return {c.target for n in neuron_ids for c in self.outgoing(n)}
