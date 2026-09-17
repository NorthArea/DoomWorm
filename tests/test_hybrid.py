"""Stage 23.3 (axis C7): hybrids — a frozen reflex brain under a trainable policy."""

from pathlib import Path

import pytest

from doomworm.candidates import (
    CandidateSpec,
    HybridBrain,
    RNNBrain,
    Trainable,
    build_candidate,
    load_candidate,
)
from doomworm.candidates.hybrid import REFLEX_CHANNELS
from doomworm.cli import main
from doomworm.learning import TrainConfig, train_candidate

REFLEX = "docs/brains/a2/worm_from_worm_evolved_random.json"
TINY = TrainConfig(train_seeds=(100,), steps=15, population=3, generations=2, workers=1)


def spec(**kw: object) -> CandidateSpec:
    """A hybrid over the A2 curriculum worm; ``init`` is the frozen reflex, not a start point."""
    return CandidateSpec(kind="hybrid", init=REFLEX, **kw)  # type: ignore[arg-type]


def test_hybrid_trains_only_the_policy_over_a_frozen_reflex() -> None:
    brain = build_candidate(spec())
    assert isinstance(brain, HybridBrain)
    assert isinstance(brain, Trainable)
    assert isinstance(brain.policy, RNNBrain)
    assert set(REFLEX_CHANNELS[:2]) <= set(brain.policy.inputs), "the policy sees the reflex"
    assert brain.n_weights == brain.policy.n_weights, "the worm's synapses are not in the genome"

    frozen = brain.reflex.get_weights()
    brain.reset()
    before = brain.act({"sensor_front": 0.8})
    brain.set_weights([v * 3.0 for v in brain.get_weights()])
    brain.reset()
    assert brain.act({"sensor_front": 0.8}) != before, "the policy decides the wheels"
    assert brain.reflex.get_weights() == pytest.approx(frozen), "the reflex stays frozen"


def test_reflex_output_reaches_the_policy() -> None:
    brain = build_candidate(spec())
    assert isinstance(brain, HybridBrain)
    brain.reset()
    brain.act({"sensor_front": 0.8})
    reflex_left, reflex_right = brain.last_reflex
    assert brain.last_channels["reflex_left"] == pytest.approx(reflex_left)
    assert brain.last_channels["reflex_right"] == pytest.approx(reflex_right)
    assert all(-1.0 <= v <= 1.0 for v in brain.last_reflex)


def test_round_trip_keeps_the_same_behaviour(tmp_path: Path) -> None:
    brain = build_candidate(spec())
    assert isinstance(brain, HybridBrain)
    brain.reset()
    expected = [brain.act({"sensor_front": 0.1 * i}) for i in range(4)]
    brain.save(tmp_path / "h.json")

    loaded = load_candidate(tmp_path / "h.json")
    assert isinstance(loaded, HybridBrain)
    loaded.reset()
    got = [loaded.act({"sensor_front": 0.1 * i}) for i in range(4)]
    assert got == pytest.approx(expected)
    assert loaded.meta["candidate"] == "hybrid"
    assert loaded.meta["reflex"].endswith("worm_from_worm_evolved_random.json")


def test_hybrid_trains_and_benchmarks_with_the_shared_harness(tmp_path: Path) -> None:
    out = tmp_path / "hybrid.json"
    result = train_candidate(spec(), TINY, out)
    assert len(result.history) == 2
    loaded = load_candidate(out)
    assert isinstance(loaded, HybridBrain)
    assert loaded.meta["layer"] == "needs"
    args = ["benchmark", "--brain", str(out), "--planner", "needs", "--test-seeds", "1"]
    assert main([*args, "--repeats", "1", "--steps", "10", "--out-dir", str(tmp_path / "b")]) == 0


def test_hybrid_needs_a_reflex_brain() -> None:
    with pytest.raises(ValueError, match="reflex"):
        build_candidate(CandidateSpec(kind="hybrid"))


def test_hybrid_carries_the_trigger_on_the_doom_task() -> None:
    brain = build_candidate(spec(maps="doom4", task="doom", sensors="ideal"))
    assert isinstance(brain, HybridBrain)
    assert brain.policy.outputs == 3, "track B: the policy also pulls the trigger"
    assert "reflex_fire" in brain.policy.inputs
    brain.reset()
    brain.act({"sensor_front": 0.5, "aim": 0.9})
    assert -1.0 <= brain.fire <= 1.0, "the policy readout, like the rnn's third unit"
