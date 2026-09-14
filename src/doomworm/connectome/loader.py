"""Load the Cook 2019 CSVs produced by ``scripts/fetch_connectome.py`` (Plan §11)."""

from __future__ import annotations

import csv
from pathlib import Path

from doomworm.connectome.model import Connection, ConnectionType, Connectome, NeuronInfo

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "connectome" / "cook2019"


def load_cook2019(data_dir: Path | str = DATA_DIR) -> Connectome:
    """Read neurons, chemical synapses and gap junctions into the internal format.

    Each gap junction pair ``(a, b)`` becomes two electrical connections
    ``a -> b`` and ``b -> a`` (one if ``a == b``), per Plan §11.
    """
    data_dir = Path(data_dir)
    neurons: list[NeuronInfo] = []
    with (data_dir / "neurons.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            meta = {k: row[k] for k in ("category", "notes") if row[k]}
            name = row["name"]
            neurons.append(NeuronInfo(id=name, name=name, type=row["type"], metadata=meta))

    connections: list[Connection] = []
    with (data_dir / "chemical.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            src, dst, w = row["source"], row["target"], float(row["weight"])
            connections.append(Connection(src, dst, w, ConnectionType.CHEMICAL))
    with (data_dir / "gap_junction.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            a, b, w = row["a"], row["b"], float(row["weight"])
            connections.append(Connection(a, b, w, ConnectionType.ELECTRICAL))
            if a != b:
                connections.append(Connection(b, a, w, ConnectionType.ELECTRICAL))
    return Connectome(neurons=neurons, connections=connections)
