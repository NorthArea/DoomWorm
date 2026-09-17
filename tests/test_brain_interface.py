"""Stage 18: Brain interface, generic episode loop, Gymnasium env, benchmark (Plan §3.3)."""

from pathlib import Path

import numpy as np
import pytest

from doomworm.candidates import Brain, ScriptedBrain, WormBrain
from doomworm.environments.gym_env import DoomwormEnv
from doomworm.environments.worlds import build_world
from doomworm.episode import run_brain_episode, run_episode
from doomworm.experiments.worm_agent import WormScenario, run_worm
from doomworm.learning import (
    BenchmarkConfig,
    RewardTracker,
    leaderboard,
    run_benchmark,
    save_result,
    write_leaderboard,
)


def straight(_: object) -> tuple[float, float]:
    return 1.0, 1.0


def test_scripted_brain_satisfies_protocol() -> None:
    b = ScriptedBrain(straight, name="straight")
    assert isinstance(b, Brain)
    assert b.act({}) == (1.0, 1.0)


def test_worm_brain_matches_the_scenario_loop() -> None:
    sc = WormScenario()
    _, trace_old, _ = run_worm(sc, seed=1001, steps=40, record_activity=False)
    brain = WormBrain.from_scenario(sc)
    assert isinstance(brain, Brain)
    world = sc.make_world(1001)
    trace_new = run_brain_episode(world, brain, 40, RewardTracker())
    assert [r.motors for r in trace_new] == pytest.approx([r.motors for r in trace_old])
    assert [(r.x, r.y) for r in trace_new] == pytest.approx([(r.x, r.y) for r in trace_old])


def test_worm_brain_save_and_load_with_overrides(tmp_path: Path) -> None:
    sc = WormScenario()
    brain = WormBrain.from_scenario(sc)
    path = tmp_path / "w.json"
    brain.save(path)
    loaded = WormBrain.from_file(path, maps="doom4", task="doom")
    assert loaded.meta["params"]["maps"] == "doom4"
    assert loaded.meta["params"]["task"] == "doom"
    assert loaded.name == "w"
    assert loaded.network.get_weights() == brain.network.get_weights()


def test_build_world_is_the_scenario_world() -> None:
    a = WormScenario(maps="doom4", task="doom").make_world(3)
    b = build_world(3, "doom4", "doom")
    assert a.walls == b.walls
    assert a.dock == b.dock
    with pytest.raises(ValueError, match="task"):
        build_world(1, "fixed", "nope")


def test_gym_env_api() -> None:
    env = DoomwormEnv(lambda seed: build_world(seed, "random", "food"), sensors="ideal", steps=20)
    obs, info = env.reset(seed=5)
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)
    assert info["seed"] == 5
    total = 0.0
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(np.array([1.0, 1.0]))
        total += reward
        if terminated or truncated:
            break
    assert truncated
    assert info["tick"] == 20
    assert total == pytest.approx(info["total_reward"])
    again, _ = env.reset(seed=5)
    assert np.allclose(again, env.reset(seed=5)[0]), "seeded reset is deterministic"
    assert "range_0" in env.channel_names
    assert isinstance(env.render(), str)


def test_gym_env_terminates_on_death() -> None:
    def factory(seed: int):  # type: ignore[no-untyped-def]
        w = build_world(seed, "fixed", "food")
        w.hunger_rate = 0.5
        return w

    env = DoomwormEnv(factory, sensors="ideal", steps=50)
    env.reset(seed=0)
    for _ in range(3):
        _, _, terminated, truncated, _ = env.step(np.zeros(2))
    assert terminated
    assert not truncated


def test_benchmark_and_leaderboard(tmp_path: Path) -> None:
    cfg = BenchmarkConfig(
        maps="random", task="food", sensors="ideal", test_seeds=(2000, 2001), steps=30, repeats=2
    )
    assert cfg.episodes == 4
    a = run_benchmark(ScriptedBrain(straight, "straight"), "straight", cfg)
    b = run_benchmark(ScriptedBrain(lambda _: (0.0, 0.0), "still"), "still", cfg)
    assert len(a.rows) == 4
    assert a.mean("distance") > b.mean("distance")
    assert set(a.summary()) >= {"reward", "coverage", "collisions", "survived"}
    save_result(a, tmp_path)
    save_result(b, tmp_path)
    table = write_leaderboard(tmp_path)
    assert table.splitlines()[0].startswith("| # | brain |")
    assert "straight" in table
    assert "still" in table
    assert (tmp_path / "leaderboard.md").exists()
    assert leaderboard([a, b]).count("\n") == 3


def test_old_loop_still_works() -> None:
    sc = WormScenario()
    world = sc.make_world(1)
    from doomworm.brain import Simulator

    trace = run_episode(world, Simulator(sc.template), sc.sensory, sc.motor, 5)
    assert len(trace) == 5
