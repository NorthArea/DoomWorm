"""PPO candidate of the A2 bake-off (Plan §20.4 #5): another class of learner.

Optional ``rl`` dependency group (torch, stable-baselines3). The policy trains
on the Gymnasium env wrapped in the same planner + needs layer the other
candidates get, cycling over the training maps; the saved policy is then a
Brain like any other and goes through the benchmark unchanged.

    broomworm ppo --timesteps 300000 --out runs/a2/ppo.json
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from broomworm.candidates.base import Wheels
from broomworm.candidates.rnn import UNBOUNDED
from broomworm.environments.gym_env import BroomwormEnv, PlannerWrapper
from broomworm.environments.sensors import PRESETS, SensorSuite
from broomworm.environments.worlds import build_world


@dataclass(frozen=True)
class PPOConfig:
    """World, layer and budget of one PPO run."""

    maps: str = "apartment"
    task: str = "clean"
    sensors: str = "vacuum"
    dangers: int = 0
    steps: int = 800
    layer: str = "needs"
    train_seeds: tuple[int, ...] = (100, 101, 102)
    timesteps: int = 300_000
    n_steps: int = 2048
    batch_size: int = 64
    seed: int = 0


def bounded_channels(sensors: str) -> list[str]:
    """Policy inputs: every channel of the preset except raw odometry."""
    return [c for c in SensorSuite(PRESETS[sensors]).channel_names if c not in UNBOUNDED]


class _SeedCycle(gym.Wrapper[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
    """Reset without an explicit seed walks round-robin over the training maps."""

    def __init__(self, env: gym.Env[np.ndarray, np.ndarray], seeds: tuple[int, ...]) -> None:
        super().__init__(env)
        self.seeds = seeds
        self.i = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is None:
            seed = self.seeds[self.i % len(self.seeds)]
            self.i += 1
        return self.env.reset(seed=seed, options=options)


def make_env(cfg: PPOConfig) -> gym.Env[np.ndarray, np.ndarray]:
    """The training environment: world -> sensors -> planner layer -> seed cycle."""
    base = BroomwormEnv(
        lambda seed: build_world(seed, cfg.maps, cfg.task, cfg.dangers),
        sensors=cfg.sensors,
        steps=cfg.steps,
        channel_names=bounded_channels(cfg.sensors),
    )
    env: gym.Env[np.ndarray, np.ndarray] = base
    if cfg.layer != "none":
        env = PlannerWrapper(base, mode=cfg.layer)
    return _SeedCycle(env, cfg.train_seeds)


def train_ppo(cfg: PPOConfig, out: Path, verbose: int = 0) -> Path:
    """Train with stable-baselines3 and save ``out`` (JSON descriptor) + ``out.zip``."""
    from stable_baselines3 import PPO

    env = make_env(cfg)
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        device="cpu",
        verbose=verbose,
    )
    from stable_baselines3.common.callbacks import CheckpointCallback

    out.parent.mkdir(parents=True, exist_ok=True)
    every = max(cfg.n_steps, cfg.timesteps // 10)
    ckpt = CheckpointCallback(every, str(out.parent / f"{out.stem}_ckpt"), name_prefix=out.stem)
    model.learn(total_timesteps=cfg.timesteps, callback=ckpt)
    model.save(out.with_suffix(".zip"))
    data = {
        "kind": "ppo",
        "model": out.with_suffix(".zip").name,
        "channel_names": bounded_channels(cfg.sensors),
        "meta": {"candidate": "ppo", "layer": cfg.layer, "train": asdict(cfg)},
    }
    out.write_text(json.dumps(data, indent=2) + "\n")
    return out


class PPOBrain:
    """A saved SB3 policy acting on the channel vector; deterministic at test time."""

    def __init__(
        self, model: Any, channel_names: list[str], name: str, meta: dict[str, Any]
    ) -> None:
        self.model = model
        self.channel_names = channel_names
        self.name = name
        self.meta = meta

    def reset(self) -> None:
        """Stateless policy (MLP)."""

    def act(self, channels: Mapping[str, float]) -> Wheels:
        """One forward pass."""
        obs = np.array([channels.get(c, 0.0) for c in self.channel_names], dtype=np.float64)
        action, _ = self.model.predict(obs, deterministic=True)
        out = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        return float(out[0]), float(out[1])

    @classmethod
    def from_file(cls, path: Path | str) -> PPOBrain:
        """Load the descriptor and the SB3 zip next to it."""
        from stable_baselines3 import PPO

        path = Path(path)
        data = json.loads(path.read_text())
        if data.get("kind") != "ppo":
            raise ValueError(f"{path} is not a ppo brain")
        model = PPO.load(path.parent / data["model"], device="cpu")
        return cls(model, list(data["channel_names"]), path.stem, dict(data.get("meta", {})))
