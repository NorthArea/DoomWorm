"""Stage 19: the animal's own noise, optional and seeded."""

from __future__ import annotations

import pytest

from wormlab.brain import Simulator
from wormlab.candidates import WormBrain
from wormlab.experiments.worm_agent import WormScenario


def scenario() -> WormScenario:
    return WormScenario(maps="doom4", task="doom", sensors="ideal")


def test_the_network_is_deterministic_unless_asked_otherwise() -> None:
    """Every published row replays: noise is off by default."""
    net = scenario().template
    a = [sum(Simulator(net).step({"ASHL": 1.0}).values()) for _ in range(5)]
    net.reset()
    b = [sum(Simulator(net).step({"ASHL": 1.0}).values()) for _ in range(5)]
    assert a == b


def test_noise_changes_the_answer_and_the_seed_pins_it() -> None:
    net = scenario().template
    quiet = [sum(Simulator(net).step({"ASHL": 1.0}).values()) for _ in range(5)]
    net.reset()
    loud = [sum(Simulator(net, noise=0.05, seed=1).step({"ASHL": 1.0}).values()) for _ in range(5)]
    net.reset()
    again = [sum(Simulator(net, noise=0.05, seed=1).step({"ASHL": 1.0}).values()) for _ in range(5)]
    assert loud != pytest.approx(quiet), "noise moves the network"
    assert loud == pytest.approx(again), "the same seed replays it"


def test_a_probabilistic_synapse_drops_signal() -> None:
    """Vesicle release: a connection transmits with probability below one."""
    net = scenario().template
    full = [sum(Simulator(net).step({"ASHL": 1.0}).values()) for _ in range(6)]
    net.reset()
    sparse = [
        sum(Simulator(net, release=0.5, seed=2).step({"ASHL": 1.0}).values()) for _ in range(6)
    ]
    assert sum(sparse) < sum(full), "half the synapses stay silent each tick"


def test_the_worm_carries_it_and_reseeds_per_episode() -> None:
    brain = WormBrain.from_scenario(scenario())
    brain.noise = 0.05
    brain.reset()
    first = [brain.act({"sensor_front": 0.3}) for _ in range(4)]
    brain.reset()
    second = [brain.act({"sensor_front": 0.3}) for _ in range(4)]
    assert first != second, "a new episode draws a new noise stream"

    quiet = WormBrain.from_scenario(scenario())
    quiet.reset()
    a = [quiet.act({"sensor_front": 0.3}) for _ in range(4)]
    quiet.reset()
    b = [quiet.act({"sensor_front": 0.3}) for _ in range(4)]
    assert a == pytest.approx(b), "without noise an episode replays exactly"
