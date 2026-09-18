"""Stage 21.4: PPO candidate through the same layer and benchmark (optional rl group)."""

from pathlib import Path

import pytest

from wormlab.cli import main

pytest.importorskip("stable_baselines3")


def test_ppo_trains_saves_and_benchmarks(tmp_path: Path) -> None:
    from wormlab.candidates import load_candidate
    from wormlab.learning.rl import PPOBrain, PPOConfig, make_env, train_ppo

    cfg = PPOConfig(train_seeds=(100, 101), steps=20, timesteps=64, n_steps=32, batch_size=32)
    env = make_env(cfg)
    obs, _ = env.reset()
    assert obs.shape[0] > 10
    out = train_ppo(cfg, tmp_path / "ppo.json")
    assert out.with_suffix(".zip").exists()
    brain = load_candidate(out)
    assert isinstance(brain, PPOBrain)
    assert brain.meta["layer"] == "needs"
    brain.reset()
    left, right = brain.act({"sensor_front": 0.5, "battery": 0.9})
    assert -1.0 <= left <= 1.0
    assert -1.0 <= right <= 1.0
    args = ["benchmark", "--brain", str(out), "--test-seeds", "1"]
    assert main([*args, "--repeats", "1", "--steps", "10", "--out-dir", str(tmp_path / "b")]) == 0
