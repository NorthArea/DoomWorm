"""Stage 8: neural activity -> wheels (Plan §14), brain ticks per env step (Plan §3.1)."""

import pytest

from doomworm.adapters import GroupMotorAdapter
from doomworm.brain import Simulator
from doomworm.connectome import (
    Connectome,
    MotorMapping,
    build_network,
    default_motor_mapping,
    default_sensory_mapping,
    load_cook2019,
)
from doomworm.environments.simple_2d import AgentState, World
from doomworm.episode import average_activity, run_episode
from doomworm.experiments import food_agent


@pytest.fixture(scope="module")
def worm() -> Connectome:
    return load_cook2019()


def make_adapter(**kwargs: float) -> GroupMotorAdapter:
    return GroupMotorAdapter(["F"], ["R"], ["TL"], ["TR"], **kwargs)


def test_default_motor_mapping_matches_plan(worm: Connectome) -> None:
    m = default_motor_mapping()
    m.validate(worm)
    assert {"AVBL", "AVBR", "PVCL", "PVCR", "VB01", "DB07"} <= set(m.forward)
    assert {"AVAL", "AVAR", "AVDL", "AVDR", "VA12", "DA09"} <= set(m.reversal)
    assert set(m.turn_left) == {"SMDDL", "SMDDR", "RIVL"}
    assert set(m.turn_right) == {"SMDVL", "SMDVR", "RIVR"}
    assert not set(m.forward) & set(m.reversal)
    assert all(worm.neuron(n).type == "motorneuron" for n in m.forward if n[:2] in ("VB", "DB"))


def test_motor_mapping_validation_and_json(tmp_path) -> None:  # type: ignore[no-untyped-def]
    m = default_motor_mapping()
    m.save(tmp_path / "motor.json")
    assert MotorMapping.load(tmp_path / "motor.json") == m
    bad = MotorMapping(forward=["AVBL"], reversal=["NOPE"], turn_left=["RIVL"], turn_right=[])
    with pytest.raises(KeyError, match="NOPE"):
        bad.validate(load_cook2019())
    with pytest.raises(ValueError, match="turn_right"):
        MotorMapping(forward=["AVBL"], reversal=["AVAL"], turn_left=["RIVL"]).validate(
            load_cook2019()
        )


def test_forward_and_reversal_drive() -> None:
    adapter = make_adapter(gain_drive=2.0, gain_turn=1.0)
    assert adapter({"F": 0.5}) == (1.0, 1.0), "clipped at 1"
    assert adapter({"F": 0.25}) == (0.5, 0.5)
    assert adapter({"R": 0.25}) == (-0.5, -0.5), "reversal = both wheels backwards"
    assert adapter({"F": 0.3, "R": 0.3}) == (0.0, 0.0)
    assert adapter({}) == (0.0, 0.0)


def test_turn_asymmetry() -> None:
    adapter = make_adapter(gain_drive=1.0, gain_turn=1.0)
    left, right = adapter({"F": 0.5, "TR": 0.2})
    assert left > right, "turn_right activity speeds up the left wheel"
    left, right = adapter({"F": 0.5, "TL": 0.2})
    assert left < right
    assert adapter({"F": 0.5, "TL": 0.2, "TR": 0.2}) == pytest.approx((0.5, 0.5))


def test_components_expose_group_means() -> None:
    adapter = GroupMotorAdapter(["a", "b"], ["c"], ["d"], ["e"], gain_drive=10.0, gain_turn=4.0)
    c = adapter.components({"a": 0.2, "b": 0.4, "c": 0.1, "e": 0.5})
    assert c["forward"] == pytest.approx(0.3)
    assert c["drive"] == pytest.approx(2.0)
    assert c["turn"] == pytest.approx(2.0)


def test_world_accepts_reverse() -> None:
    w = World(agent=AgentState(x=10.0, y=10.0), speed=0.5)
    w.step(-1.0, -1.0)
    assert w.agent.x == pytest.approx(9.5)
    w = World(agent=AgentState(x=10.0, y=10.0), speed=0.5)
    w.step(5.0, -3.0)  # clamped to (1, -1): rotate right in place
    assert (w.agent.x, w.agent.y) == pytest.approx((10.0, 10.0))
    assert w.agent.heading == pytest.approx(-1.0)


def test_average_activity() -> None:
    assert average_activity([{"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}]) == {"a": 0.5, "b": 0.5}
    single = {"a": 0.3}
    assert average_activity([single]) is single


def test_brain_steps_window_in_episode() -> None:
    world, sim, sensory, motor = food_agent.build_scenario()
    trace = run_episode(world, sim, sensory, motor, steps=10, brain_steps=4, record_activity=True)
    assert sim.time == 40
    assert len(trace) == 10
    assert trace[0].activity is not None
    with pytest.raises(ValueError, match="brain_steps"):
        run_episode(world, sim, sensory, motor, steps=1, brain_steps=0)


def test_front_touch_produces_reversal(worm: Connectome) -> None:
    """AVM/FLP -> AVA/AVD: the loaded network drives the wheels backwards on touch."""
    net = build_network(worm)
    sim = Simulator(net)
    sensory = default_sensory_mapping().to_adapter()
    m = default_motor_mapping()
    motor = GroupMotorAdapter(m.forward, m.reversal, m.turn_left, m.turn_right)
    currents = sensory({"sensor_front": 1.0})
    activity = average_activity([sim.step(currents) for _ in range(5)])
    c = motor.components(activity)
    assert c["reversal"] > c["forward"] > 0.0
    left, right = motor(activity)
    assert left < 0.0
    assert right < 0.0
    assert left == pytest.approx(right, abs=0.05), "front input is symmetric"


def test_food_activates_head_turning_neurons(worm: Connectome) -> None:
    net = build_network(worm)
    sim = Simulator(net)
    sensory = default_sensory_mapping().to_adapter()
    m = default_motor_mapping()
    motor = GroupMotorAdapter(m.forward, m.reversal, m.turn_left, m.turn_right)
    currents = sensory({"food_front": 1.0})
    activity = average_activity([sim.step(currents) for _ in range(5)])
    c = motor.components(activity)
    assert c["turn_left"] > c["forward"]
    assert c["turn_right"] > c["reversal"]
