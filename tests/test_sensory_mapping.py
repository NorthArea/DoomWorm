"""Stage 7: world channels -> sensory neurons (Plan §13)."""

from pathlib import Path

import pytest

from broomworm.brain import Simulator, stimulate
from broomworm.connectome import (
    Connectome,
    SensoryMapping,
    build_network,
    default_sensory_mapping,
    load_cook2019,
)
from broomworm.environments.simple_2d import AgentState, Food, Obstacle, World


@pytest.fixture(scope="module")
def worm() -> Connectome:
    return load_cook2019()


# `hunger` does not describe the outside world: it is an internal state and goes to
# the neurosecretory pair NSM, not to a sensory neuron.
INTERNAL_ROUTES = {("hunger", "NSML"), ("hunger", "NSMR")}


def test_default_mapping_neurons_exist_and_are_sensory(worm: Connectome) -> None:
    m = default_sensory_mapping()
    m.validate(worm)
    for channel in m.channels():
        for neuron, _ in m.targets(channel):
            if (channel, neuron) in INTERNAL_ROUTES:
                assert worm.neuron(neuron).type == "interneuron", (channel, neuron)
                continue
            assert worm.neuron(neuron).type == "sensory", (channel, neuron)


def test_default_mapping_is_lateralised() -> None:
    m = default_sensory_mapping()
    left = {n for n, _ in m.targets("sensor_left")}
    right = {n for n, _ in m.targets("sensor_right")}
    front = {n for n, _ in m.targets("sensor_front")}
    assert left == {"ALML", "FLPL"}
    assert right == {"ALMR", "FLPR"}
    assert front == {"AVM", "FLPL", "FLPR"}, "front drives both members of the pair"
    assert {n for n, _ in m.targets("food_front")} == {n for n, _ in m.targets("food_left")} | {
        n for n, _ in m.targets("food_right")
    }
    assert {n for n, _ in m.targets("hunger")} == {"NSML", "NSMR", "ASIL", "ASIR"}


def test_validate_rejects_unknown_neuron(worm: Connectome) -> None:
    m = SensoryMapping().add("sensor_front", ["AVM", "NOPE"])
    with pytest.raises(KeyError, match="NOPE"):
        m.validate(worm)


def test_adapter_sums_gains_and_channels() -> None:
    m = SensoryMapping().add("a", ["N1", "N2"], gain=2.0).add("b", ["N2"], gain=0.5)
    m.tonic["N3"] = 1.0
    adapter = m.to_adapter()
    currents = adapter({"a": 0.5, "b": 1.0})
    assert currents == {"N1": 1.0, "N2": 1.5, "N3": 1.0}
    assert adapter.neurons == {"N1", "N2", "N3"}


def test_json_round_trip(tmp_path: Path) -> None:
    m = default_sensory_mapping()
    m.tonic["AVBL"] = 0.2
    m.save(tmp_path / "map.json")
    back = SensoryMapping.load(tmp_path / "map.json")
    assert back == m
    assert "sensor_left" in back.describe()


def test_world_observation_drives_named_neurons() -> None:
    world = World(
        agent=AgentState(x=10.0, y=10.0),
        obstacles=[Obstacle(x=13.0, y=10.0, radius=1.0)],  # ahead, reading 0.5
        foods=[Food(x=10.0, y=14.0)],  # left, reading 0.25
    )
    currents = default_sensory_mapping().to_adapter()(world.observe().as_channels())
    assert currents["AVM"] == pytest.approx(0.5)
    assert currents["FLPL"] == pytest.approx(0.5)
    assert currents["ALML"] == 0.0
    assert currents["AWCL"] == pytest.approx(0.25)
    assert currents["AWCR"] == 0.0
    assert currents["ASHL"] == 0.0


def test_front_touch_reaches_reversal_command_neurons(worm: Connectome) -> None:
    """ALM/AVM/FLP -> AVA/AVD is the classic touch-escape circuit."""
    net = build_network(worm)
    adapter = default_sensory_mapping().to_adapter()
    trace = stimulate(net, adapter({"sensor_front": 1.0}), hold=10, settle=10)
    first = trace.first_active(0.005)
    assert first["AVM"] == 0
    assert 0 < first["AVAL"] <= 3
    assert 0 < first["AVDL"] <= 3
    assert trace.peak()["AVAL"] > 0.0


def test_food_smell_reaches_aiy(worm: Connectome) -> None:
    """AWC -> AIY is the best-known chemotaxis synapse."""
    net = build_network(worm)
    adapter = default_sensory_mapping().to_adapter()
    trace = stimulate(net, adapter({"food_front": 1.0}), hold=10, settle=10)
    first = trace.first_active(0.005)
    assert first["AIYL"] == 1
    assert first["AIYR"] == 1


def test_mapping_runs_in_closed_loop(worm: Connectome) -> None:
    net = build_network(worm)
    adapter = default_sensory_mapping().to_adapter()
    world = World(agent=AgentState(x=3.0, y=10.0), obstacles=[Obstacle(x=6.0, y=10.0, radius=1.0)])
    sim = Simulator(net)
    activity = {}
    for _ in range(5):
        activity = sim.step(adapter(world.observe().as_channels()))
    assert activity["AVM"] > 0.0
