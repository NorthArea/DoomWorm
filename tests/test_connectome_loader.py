"""Stage 5: Cook 2019 loader, internal format, network build (Plan §11, §43)."""

import pytest

from doomworm.brain import Simulator
from doomworm.connectome import (
    GABA_NEURONS,
    Connection,
    ConnectionType,
    Connectome,
    NeuronInfo,
    build_network,
    load_cook2019,
)


@pytest.fixture(scope="module")
def worm() -> Connectome:
    return load_cook2019()


def test_connectome_loader(worm: Connectome) -> None:
    assert len(worm) == 302
    types = {t: len(worm.by_type(t)) for t in ("sensory", "interneuron", "motorneuron", "neuron")}
    assert types == {"sensory": 83, "interneuron": 91, "motorneuron": 126, "neuron": 2}
    assert sum(types.values()) == 302
    assert len(worm.of_type(ConnectionType.CHEMICAL)) == 3709
    assert len(worm.of_type(ConnectionType.ELECTRICAL)) == 2 * 1105 - 14  # 14 self entries


def test_known_connections(worm: Connectome) -> None:
    chem = {(c.source, c.target): c.weight for c in worm.of_type(ConnectionType.CHEMICAL)}
    assert chem[("ASHL", "AVAL")] == 15.0
    assert chem[("AWCL", "AIYL")] == 22.0
    gap = {(c.source, c.target): c.weight for c in worm.of_type(ConnectionType.ELECTRICAL)}
    assert gap[("AVAL", "VA02")] == gap[("VA02", "AVAL")] == 13.0


def test_electrical_connections_are_symmetric(worm: Connectome) -> None:
    gap = {(c.source, c.target): c.weight for c in worm.of_type(ConnectionType.ELECTRICAL)}
    assert all(gap.get((b, a)) == w for (a, b), w in gap.items())


def test_neuron_lookup_and_neighbours(worm: Connectome) -> None:
    assert worm.neuron("ASHL").type == "sensory"
    assert "AVAL" in worm.downstream(["ASHL"])
    assert worm.neuron("CANL").type == "neuron"
    assert "nope" not in worm


def test_model_rejects_dangling_connection() -> None:
    n = [NeuronInfo("a", "a", "sensory")]
    with pytest.raises(KeyError):
        Connectome(n, [Connection("a", "b", 1.0, ConnectionType.CHEMICAL)])


def test_build_network_shape_and_signs(worm: Connectome) -> None:
    net = build_network(worm, gain=1.0)
    assert len(net.neurons) == 302
    assert len(net.synapses) == len(worm.connections)
    assert all(n.graded for n in net.neurons.values())

    for s in net.synapses:
        if s.kind == "electrical":
            assert s.weight > 0.0
        elif s.source in GABA_NEURONS:
            assert s.weight < 0.0
        else:
            assert s.weight > 0.0

    for nid in net.neurons:
        total = sum(abs(s.weight) for s in net.incoming(nid))
        assert total == pytest.approx(1.0) or not net.incoming(nid)


def test_build_network_gain_scales_weights(worm: Connectome) -> None:
    a = build_network(worm, gain=1.0).get_weights()
    b = build_network(worm, gain=2.5).get_weights()
    assert b == pytest.approx([w * 2.5 for w in a])


def test_stimulus_propagates_downstream(worm: Connectome) -> None:
    net = build_network(worm, gain=2.0, decay=0.5)
    sim = Simulator(net)
    for _ in range(5):
        activity = sim.step({"ASHL": 1.0})
    assert activity["ASHL"] == 1.0
    assert activity["AVAL"] > 0.0
    assert activity["AVAR"] > 0.0
