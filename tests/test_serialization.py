"""Brain JSON format (Plan §40)."""

import json
from pathlib import Path

import pytest

from doomworm.brain import Network, Neuron, Synapse, load_brain, network_from_dict, save_brain
from doomworm.brain.serialization import FORMAT_VERSION, network_to_dict


def make_net() -> Network:
    net = Network()
    net.add_neuron(Neuron("a", threshold=0.5, decay=0.2))
    net.add_neuron(Neuron("b"))
    net.add_synapse(Synapse("a", "b", weight=-0.75))
    net.add_synapse(Synapse("b", "a", weight=0.25, kind="electrical"))
    return net


def test_round_trip_dict() -> None:
    net = make_net()
    net.neurons["a"].potential = 3.0  # transient state must not be serialised
    d = network_to_dict(net)
    assert d["format"] == FORMAT_VERSION
    assert d["neurons"][0] == {"id": "a", "threshold": 0.5, "decay": 0.2}
    assert d["synapses"][1] == {"source": "b", "target": "a", "weight": 0.25, "kind": "electrical"}
    clone = network_from_dict(d)
    assert network_to_dict(clone) == d
    assert clone.neurons["a"].potential == 0.0


def test_save_and_load(tmp_path: Path) -> None:
    path = tmp_path / "brain.json"
    save_brain(path, make_net(), meta={"seed": 7, "stage": 4})
    loaded, meta = load_brain(path)
    assert meta == {"seed": 7, "stage": 4}
    assert loaded.get_weights() == [-0.75, 0.25]
    assert json.loads(path.read_text())["format"] == FORMAT_VERSION


def test_unknown_format_rejected() -> None:
    d = network_to_dict(make_net())
    d["format"] = FORMAT_VERSION + 1
    with pytest.raises(ValueError, match="format"):
        network_from_dict(d)


def test_weights_get_set_and_reset() -> None:
    net = make_net()
    net.set_weights([1.5, -2.0])
    assert net.get_weights() == [1.5, -2.0]
    with pytest.raises(ValueError, match="2 weights"):
        net.set_weights([1.0])
    net.neurons["a"].potential = 0.9
    net.neurons["a"].activity = 1.0
    net.reset()
    assert (net.neurons["a"].potential, net.neurons["a"].activity) == (0.0, 0.0)
