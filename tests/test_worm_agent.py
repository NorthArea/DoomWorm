"""Stage 9: the untrained connectome drives the 2D agent (Plan §15)."""

import json
import math
from pathlib import Path

import pytest

from wormlab.brain import Simulator, save_brain
from wormlab.cli import main as cli_main
from wormlab.connectome import Connectome, load_cook2019
from wormlab.episode import average_activity
from wormlab.experiments.episode import save_log
from wormlab.experiments.worm_agent import SCENARIO_NAME, WormScenario, run_worm, summarise
from wormlab.learning import evaluate


@pytest.fixture(scope="module")
def worm() -> Connectome:
    return load_cook2019()


@pytest.fixture(scope="module")
def scenario(worm: Connectome) -> WormScenario:
    return WormScenario(connectome=worm)


def test_scenario_wires_mappings_and_tonic(scenario: WormScenario) -> None:
    assert scenario.sensory.tonic == {"AVBL": 0.1, "AVBR": 0.1}
    assert {"AVM", "AWCL", "NSML", "ASHL"} <= scenario.sensory.neurons
    assert scenario.motor.forward[:2] == ["AVBL", "AVBR"]
    assert scenario.brain_steps == 5
    assert scenario.n_weights == len(scenario.template.synapses)
    assert scenario.params["obstacle_gain"] == 3.0
    assert scenario.sensory({"sensor_front": 0.5})["AVM"] == pytest.approx(1.5)


def test_untrained_worm_moves_and_is_deterministic(scenario: WormScenario) -> None:
    world, trace, tracker = run_worm(scenario, seed=1001, steps=100)
    summary = summarise(trace, world)
    assert summary["distance"] > 1.0
    assert summary["mean_left"] > 0.0
    assert summary["mean_right"] > 0.0
    assert tracker.total == pytest.approx(sum(r.reward for r in trace))

    _, trace2, _ = run_worm(scenario, seed=1001, steps=100)
    assert [(r.x, r.y, r.motors) for r in trace] == [(r.x, r.y, r.motors) for r in trace2]


def test_touch_reduces_forward_drive(scenario: WormScenario) -> None:
    """Sign-correct but weak: touch lowers drive, it does not reverse it (untrained)."""

    def drive(channels: dict[str, float]) -> float:
        scenario.template.reset()
        sim = Simulator(scenario.template)
        currents = scenario.sensory(channels)
        activity = average_activity([sim.step(currents) for _ in range(scenario.brain_steps * 4)])
        return scenario.motor.components(activity)["drive"]

    free = drive({})
    touched = drive({"sensor_front": 1.0})
    assert free > 0.0
    assert touched < free
    assert touched > 0.0, "documented: no AVA/AVB mutual inhibition in the data, no reversal"


def test_log_has_required_fields(scenario: WormScenario, tmp_path: Path) -> None:
    _, trace, _ = run_worm(scenario, seed=1000, steps=5)
    log = tmp_path / "log.jsonl"
    save_log(trace, log, top=5)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 5
    required = {
        "tick", "x", "y", "heading", "obstacle", "food", "hunger", "motors",
        "reward", "collided", "ate", "starved", "active_neurons",
    }  # fmt: skip
    assert required <= set(rows[0])
    assert len(rows[-1]["active_neurons"]) <= 5
    assert all(v > 0 for v in rows[-1]["active_neurons"].values())


def test_worm_brain_round_trips_through_play(scenario: WormScenario, tmp_path: Path) -> None:
    path = tmp_path / "worm.json"
    save_brain(path, scenario.template, meta={"scenario": SCENARIO_NAME, "params": scenario.params})
    assert cli_main(["play", "--brain", str(path), "--seed", "1001", "--steps", "20"]) == 0


def test_worm_scenario_fits_the_learning_protocol(scenario: WormScenario) -> None:
    weights = scenario.template.get_weights()
    result = evaluate(scenario, weights, seeds=(1001,), steps=30)
    assert math.isfinite(result.fitness)
    assert result.ticks == 30
