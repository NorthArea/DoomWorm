"""Stage 21.3: a small recurrent network from scratch behind the same interface."""

from pathlib import Path

import pytest

from doomworm.brains import CandidateSpec, RNNBrain, Trainable, build_candidate, load_candidate
from doomworm.cli import main
from doomworm.learning import TrainConfig, train_candidate

TINY = TrainConfig(train_seeds=(100,), steps=15, population=3, generations=2, workers=1)


def test_rnn_is_a_small_trainable_brain_on_bounded_channels() -> None:
    brain = build_candidate(CandidateSpec("rnn"))
    assert isinstance(brain, RNNBrain)
    assert isinstance(brain, Trainable)
    assert "odom_x" not in brain.inputs
    assert {"sensor_front", "bumper_left", "battery", "target_left"} <= set(brain.inputs)
    assert 1000 < brain.n_weights < 6000, "smaller than the worm's 5905 synapses"
    brain.reset()
    a = brain.act({"sensor_front": 0.9})
    brain.reset()
    b = brain.act({"sensor_front": 0.9})
    assert a == b, "deterministic"
    assert all(-1.0 <= w <= 1.0 for w in a)
    brain.reset()
    left, right = brain.act({})
    assert left > 0.2
    assert right > 0.2, "untrained: drives forward (tonic bias), does not stand still"


def test_weights_round_trip_and_change_behaviour(tmp_path: Path) -> None:
    brain = build_candidate(CandidateSpec("rnn", variant_seed=2))
    assert isinstance(brain, RNNBrain)
    w = brain.get_weights()
    brain.reset()
    before = brain.act({"sensor_left": 0.8})
    brain.set_weights([v * 3.0 for v in w])
    brain.reset()
    assert brain.act({"sensor_left": 0.8}) != before
    brain.save(tmp_path / "r.json")
    loaded = load_candidate(tmp_path / "r.json")
    assert isinstance(loaded, RNNBrain)
    assert loaded.get_weights() == pytest.approx(brain.get_weights())
    assert loaded.inputs == brain.inputs


def test_rnn_trains_and_benchmarks_with_the_shared_harness(tmp_path: Path) -> None:
    out = tmp_path / "rnn.json"
    result = train_candidate(CandidateSpec("rnn"), TINY, out)
    assert len(result.history) == 2
    loaded = load_candidate(out)
    assert isinstance(loaded, RNNBrain)
    assert loaded.meta["layer"] == "needs"
    args = ["benchmark", "--brain", str(out), "--planner", "needs", "--test-seeds", "1"]
    assert main([*args, "--repeats", "1", "--steps", "10", "--out-dir", str(tmp_path / "b")]) == 0
