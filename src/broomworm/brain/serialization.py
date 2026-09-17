"""Brain JSON format (Plan §40).

```
{
  "format": 1,
  "neurons":  [{"id", "threshold", "decay", "graded"}],
  "synapses": [{"source", "target", "weight", "kind"}],
  "meta":     {...}
}
```

Transient state (potential, activity) is never stored. Bump
``FORMAT_VERSION`` on any incompatible change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from broomworm.brain.network import Network
from broomworm.brain.neuron import Neuron
from broomworm.brain.synapse import Synapse

FORMAT_VERSION = 1


def network_to_dict(net: Network, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Serialise topology and weights."""
    return {
        "format": FORMAT_VERSION,
        "neurons": [
            {"id": n.id, "threshold": n.threshold, "decay": n.decay, "graded": n.graded}
            for n in net.neurons.values()
        ],
        "synapses": [
            {"source": s.source, "target": s.target, "weight": s.weight, "kind": s.kind}
            for s in net.synapses
        ],
        "meta": dict(meta or {}),
    }


def network_from_dict(data: dict[str, Any]) -> Network:
    """Rebuild a network; raises on unknown format."""
    version = data.get("format")
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported brain format {version!r}, expected {FORMAT_VERSION}")
    net = Network()
    for n in data["neurons"]:
        graded = bool(n.get("graded", False))
        net.add_neuron(Neuron(n["id"], threshold=n["threshold"], decay=n["decay"], graded=graded))
    for s in data["synapses"]:
        kind = s.get("kind", "chemical")
        net.add_synapse(Synapse(s["source"], s["target"], weight=s["weight"], kind=kind))
    return net


def save_brain(path: Path | str, net: Network, meta: dict[str, Any] | None = None) -> None:
    """Write a brain JSON file, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(network_to_dict(net, meta), indent=2) + "\n")


def load_brain(path: Path | str) -> tuple[Network, dict[str, Any]]:
    """Read a brain JSON file; return the network and its metadata."""
    data = json.loads(Path(path).read_text())
    return network_from_dict(data), dict(data.get("meta", {}))
