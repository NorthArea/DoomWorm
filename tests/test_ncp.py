"""Stage 21.5: Neural Circuit Policy candidate, evolved like the others (optional rl group)."""

from pathlib import Path

import pytest

from doomworm.brains import CandidateSpec, Trainable, build_candidate, load_candidate
from doomworm.cli import main
from doomworm.learning import TrainConfig, train_candidate

pytest.importorskip("ncps")

TINY = TrainConfig(train_seeds=(100,), steps=15, population=3, generations=2, workers=1)


def test_ncp_is_a_trainable_brain_with_sparse_wiring() -> None:
    from doomworm.brains.ncp import NCPBrain

    brain = build_candidate(CandidateSpec("ncp"))
    assert isinstance(brain, NCPBrain)
    assert isinstance(brain, Trainable)
    assert "odom_x" not in brain.inputs
    assert brain.wiring.units == 32
    assert brain.n_weights == len(brain.get_weights()) > 1000
    brain.reset()
    a = brain.act({"sensor_front": 0.9})
    brain.reset()
    assert brain.act({"sensor_front": 0.9}) == a
    assert all(-1.0 <= w <= 1.0 for w in a)


def test_ncp_round_trips_and_trains(tmp_path: Path) -> None:
    from doomworm.brains.ncp import NCPBrain

    brain = build_candidate(CandidateSpec("ncp", variant_seed=4))
    brain.set_weights([v * 2.0 for v in brain.get_weights()])
    brain.save(tmp_path / "n.json")
    loaded = load_candidate(tmp_path / "n.json")
    assert isinstance(loaded, NCPBrain)
    assert loaded.get_weights() == pytest.approx(brain.get_weights(), abs=1e-6)
    out = tmp_path / "ncp.json"
    result = train_candidate(CandidateSpec("ncp"), TINY, out)
    assert len(result.history) == 2
    args = ["benchmark", "--brain", str(out), "--planner", "needs", "--test-seeds", "1"]
    assert main([*args, "--repeats", "1", "--steps", "10", "--out-dir", str(tmp_path / "b")]) == 0
