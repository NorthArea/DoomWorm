"""The NumPy simulator must reproduce the per-neuron reference dynamics exactly."""

import numpy as np
import pytest

from wormlab.brain import Network, Neuron, Simulator, Synapse


def reference_step(net: Network, state: dict[str, tuple[float, float]], inputs: dict[str, float]):  # type: ignore[no-untyped-def]
    """Plain-Python semantics of Neuron.integrate/update for one tick."""
    prev = {nid: a for nid, (_, a) in state.items()}
    new = {}
    for nid, n in net.neurons.items():
        current = inputs.get(nid, 0.0) + sum(s.weight * prev[s.source] for s in net.incoming(nid))
        p = state[nid][0] * (1.0 - n.decay) + current
        if n.graded:
            a = min(1.0, max(0.0, p / n.threshold))
        elif p >= n.threshold:
            a, p = 1.0, 0.0
        else:
            a = 0.0
        new[nid] = (p, a)
    return new


def random_network(rng: np.random.Generator, n: int = 12, m: int = 40) -> Network:
    net = Network()
    for i in range(n):
        net.add_neuron(
            Neuron(
                f"n{i}",
                threshold=float(rng.uniform(0.5, 1.5)),
                decay=float(rng.uniform(0.0, 1.0)),
                graded=bool(rng.integers(0, 2)),
            )
        )
    ids = list(net.neurons)
    for _ in range(m):
        a, b = rng.choice(ids, 2)  # duplicates and self-loops allowed
        net.add_synapse(Synapse(str(a), str(b), weight=float(rng.normal(0, 0.8))))
    return net


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_reference_dynamics(seed: int) -> None:
    rng = np.random.default_rng(seed)
    net = random_network(rng)
    sim = Simulator(net)
    state = {nid: (0.0, 0.0) for nid in net.neurons}
    for _ in range(25):
        inputs = {nid: float(rng.uniform(0, 1.5)) for nid in rng.choice(list(net.neurons), 4)}
        state = reference_step(net, state, inputs)
        activity = sim.step(inputs)
        for nid, (p, a) in state.items():
            assert activity[nid] == pytest.approx(a, abs=1e-12)
            assert net.neurons[nid].potential == pytest.approx(p, abs=1e-12)
            assert net.neurons[nid].activity == pytest.approx(a, abs=1e-12)


def test_state_written_back_and_reset_respected() -> None:
    net = Network()
    net.add_neuron(Neuron("a", threshold=1.0, decay=0.0, graded=True))
    sim = Simulator(net)
    sim.step({"a": 0.4})
    assert net.neurons["a"].potential == pytest.approx(0.4)
    assert net.activities() == {"a": pytest.approx(0.4)}
    net.reset()
    sim2 = Simulator(net)
    assert sim2.step({}) == {"a": 0.0}


def test_duplicate_synapses_add_up() -> None:
    net = Network()
    net.add_neuron(Neuron("a", graded=True, decay=1.0))
    net.add_neuron(Neuron("b", graded=True, decay=1.0))
    net.add_synapse(Synapse("a", "b", weight=0.3))
    net.add_synapse(Synapse("a", "b", weight=0.2))
    sim = Simulator(net)
    sim.step({"a": 1.0})
    assert sim.step({"a": 1.0})["b"] == pytest.approx(0.5)
