"""Stage 21.2: one training harness for every trainable candidate (Plan §20.4)."""

from pathlib import Path

import numpy as np
import pytest

from doomworm.candidates import CANDIDATES, CandidateSpec, Trainable, WormBrain, build_candidate
from doomworm.cli import main
from doomworm.learning import TrainConfig, fitness_of, train_candidate

TINY = TrainConfig(train_seeds=(100,), steps=15, population=3, generations=2, workers=1)


@pytest.mark.parametrize("kind", CANDIDATES)
def test_every_candidate_is_a_trainable_brain(kind: str) -> None:
    if kind == "ncp":
        pytest.importorskip("ncps")
    # the hybrid is the one candidate that is built on top of another brain (axis C7)
    init = "docs/brains/a2/worm_from_worm_evolved_random.json" if kind == "hybrid" else None
    brain = build_candidate(CandidateSpec(kind, init=init))
    assert isinstance(brain, Trainable)
    assert brain.n_weights == len(brain.get_weights()) > 1000
    assert brain.meta["candidate"] == kind
    brain.reset()
    left, right = brain.act({"battery": 1.0})
    assert -1.0 <= left <= 1.0
    assert -1.0 <= right <= 1.0


def test_control_topologies_differ_from_the_connectome() -> None:
    real = build_candidate(CandidateSpec("worm"))
    dense = build_candidate(CandidateSpec("worm_dense"))
    random = build_candidate(CandidateSpec("worm_random", variant_seed=3))
    assert dense.n_weights > real.n_weights
    assert random.n_weights == real.n_weights
    assert random.get_weights() != real.get_weights()


def test_fitness_is_deterministic_and_depends_on_weights() -> None:
    spec = CandidateSpec("worm")
    w = np.asarray(build_candidate(spec).get_weights())
    a, b = fitness_of(spec, TINY, w), fitness_of(spec, TINY, w)
    assert a == b
    assert fitness_of(spec, TINY, w * 0.0) != a


def test_train_saves_a_brain_the_benchmark_loads(tmp_path: Path) -> None:
    out = tmp_path / "worm.json"
    spec = CandidateSpec("worm_random", variant_seed=1)
    seen: list[bool] = []
    result = train_candidate(spec, TINY, out, on_generation=lambda _: seen.append(out.exists()))
    assert seen == [True, True], "a checkpoint is written after every generation"
    assert len(result.history) == 2
    lines = out.with_suffix(".csv").read_text().splitlines()
    assert lines[0] == "generation,best,mean,worst"
    assert len(lines) == 3
    loaded = WormBrain.from_file(out)
    assert loaded.n_weights == build_candidate(spec).n_weights, "control topology restored"
    assert loaded.get_weights() == pytest.approx(list(result.best_weights))
    assert loaded.meta["candidate"] == "worm_random"
    assert loaded.meta["layer"] == "needs"
    args = ["benchmark", "--brain", str(out), "--planner", "needs", "--test-seeds", "1"]
    assert main([*args, "--repeats", "1", "--steps", "10", "--out-dir", str(tmp_path / "b")]) == 0


def test_parallel_evaluation_matches_serial(tmp_path: Path) -> None:
    spec = CandidateSpec("worm")
    serial = train_candidate(spec, TINY, tmp_path / "s.json")
    parallel = train_candidate(
        spec, TrainConfig(**{**TINY.__dict__, "workers": 2}), tmp_path / "p.json"
    )
    assert [g.best for g in serial.history] == [g.best for g in parallel.history]


def test_evolve_cli(tmp_path: Path) -> None:
    out = tmp_path / "w.json"
    args = ["evolve", "--candidate", "worm", "--train-seeds", "1", "--steps", "10"]
    assert main([*args, "--population", "2", "--generations", "1", "--out", str(out)]) == 0
    assert out.exists()


def test_seed_summary_aggregates_over_directories(tmp_path: Path) -> None:
    from doomworm.candidates import ScriptedBrain
    from doomworm.learning import BenchmarkConfig, run_benchmark, save_result
    from doomworm.learning.summary import collect, main, summary_table

    cfg = BenchmarkConfig(maps="random", task="food", sensors="ideal", test_seeds=(1,), steps=10)

    def constant(speed: float) -> ScriptedBrain:
        return ScriptedBrain(lambda _c: (speed, speed), "go")

    for seed_dir, speed in (("s0", 1.0), ("s1", 0.5)):
        d = tmp_path / seed_dir
        save_result(run_benchmark(constant(speed), "go", cfg), d)
        save_result(run_benchmark(ScriptedBrain(lambda _c: (0.0, 0.0), "still"), "still", cfg), d)
    groups = collect([tmp_path / "s0", tmp_path / "s1"])
    assert {k: len(v) for k, v in groups.items()} == {"go": 2, "still": 2}
    table = summary_table(groups)
    assert table.count("\n") == 3
    assert "| 2 |" in table
    assert main([str(tmp_path / "s0"), str(tmp_path / "s1"), "--out", str(tmp_path / "t.md")]) == 0
    assert (tmp_path / "t.md").read_text().startswith("| # | brain | seeds |")
