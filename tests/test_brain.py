"""Stage 0: Neuron, Synapse, Network, Simulator (Plan §6, §43)."""

import pytest

from wormlab.brain import Network, Neuron, Simulator, Synapse


def make_chain() -> Network:
    """INPUT -> N1 -> OUTPUT with unit weights and unit thresholds."""
    net = Network()
    for name in ("INPUT", "N1", "OUTPUT"):
        net.add_neuron(Neuron(name, threshold=1.0, decay=0.5))
    net.add_synapse(Synapse("INPUT", "N1", weight=1.0))
    net.add_synapse(Synapse("N1", "OUTPUT", weight=1.0))
    return net


# --- Neuron -----------------------------------------------------------------


def test_neuron_activation() -> None:
    n = Neuron("n", threshold=1.0, decay=0.0)
    n.integrate(0.5)
    assert n.update() == 0.0
    assert n.potential == pytest.approx(0.5)
    n.integrate(0.5)
    assert n.update() == 1.0
    assert n.potential == 0.0, "potential resets after firing"


def test_neuron_leaks() -> None:
    n = Neuron("n", threshold=10.0, decay=0.5)
    n.integrate(1.0)
    n.update()
    n.integrate(0.0)
    n.update()
    assert n.potential == pytest.approx(0.5)


def test_neuron_rejects_bad_params() -> None:
    with pytest.raises(ValueError, match="decay"):
        Neuron("n", decay=1.5)
    with pytest.raises(ValueError, match="threshold"):
        Neuron("n", threshold=0.0)


# --- Synapse ----------------------------------------------------------------


def test_synapse_signal() -> None:
    s = Synapse("a", "b", weight=0.5)
    assert s.transmit(1.0) == pytest.approx(0.5)
    assert s.transmit(0.0) == 0.0


# --- Network ----------------------------------------------------------------


def test_network_rejects_unknown_ids() -> None:
    net = Network()
    net.add_neuron(Neuron("a"))
    with pytest.raises(KeyError):
        net.add_synapse(Synapse("a", "missing"))
    with pytest.raises(ValueError, match="already"):
        net.add_neuron(Neuron("a"))


def test_network_incoming() -> None:
    net = make_chain()
    assert [s.source for s in net.incoming("OUTPUT")] == ["N1"]
    assert net.incoming("INPUT") == []


# --- Simulator --------------------------------------------------------------


def test_network_step() -> None:
    sim = Simulator(make_chain())
    activity = sim.step({"INPUT": 1.0})
    assert activity == {"INPUT": 1.0, "N1": 0.0, "OUTPUT": 0.0}
    activity = sim.step()
    assert activity["N1"] == 1.0
    assert activity["OUTPUT"] == 0.0
    activity = sim.step()
    assert activity["OUTPUT"] == 1.0
    assert sim.time == 3


def test_step_rejects_unknown_input() -> None:
    sim = Simulator(make_chain())
    with pytest.raises(KeyError):
        sim.step({"nope": 1.0})


def test_acceptance_input_zero_output_inactive() -> None:
    sim = Simulator(make_chain())
    history = sim.run(10, {"INPUT": 0.0})
    assert all(a["OUTPUT"] == 0.0 for a in history)


def test_acceptance_input_one_output_active() -> None:
    sim = Simulator(make_chain())
    history = sim.run(10, {"INPUT": 1.0})
    assert any(a["OUTPUT"] == 1.0 for a in history)


def test_simulation_is_deterministic() -> None:
    a = Simulator(make_chain()).run(20, {"INPUT": 0.7})
    b = Simulator(make_chain()).run(20, {"INPUT": 0.7})
    assert a == b


# --- Graded activity (Plan §2.3, stage 5+) -------------------------------------


def test_graded_neuron_activity_is_potential_over_threshold() -> None:
    n = Neuron("g", threshold=2.0, decay=0.0, graded=True)
    n.integrate(0.5)
    assert n.update() == pytest.approx(0.25)
    assert n.potential == pytest.approx(0.5), "no reset in graded mode"
    n.integrate(3.0)
    assert n.update() == 1.0, "clipped at 1"
    n.integrate(-10.0)
    assert n.update() == 0.0, "clipped at 0"


def test_graded_chain_passes_fractional_signal() -> None:
    net = Network()
    for name in ("A", "B"):
        net.add_neuron(Neuron(name, threshold=1.0, decay=1.0, graded=True))
    net.add_synapse(Synapse("A", "B", weight=0.5))
    sim = Simulator(net)
    sim.step({"A": 0.6})
    activity = sim.step({"A": 0.6})
    assert activity["A"] == pytest.approx(0.6)
    assert activity["B"] == pytest.approx(0.3)
