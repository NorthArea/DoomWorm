"""The engineered layer as a Gymnasium wrapper, so PPO meets what the others meet.

Every candidate of this track is benchmarked wrapped in :class:`PlannerLayer`.
A policy trained by PPO learns inside a Gymnasium environment instead, so the
same layer has to sit there too: the planner's virtual ``target_*`` gradient
replaces the raw one in the observation, and while the robot is parked on the
dock the action is overridden with stopped wheels. Without this the PPO row is
measured under different conditions than the rows it is compared with.

Importing this module registers the wrapper for ``PPOConfig.layer``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np

from broomworm.layer import PlannerLayer
from wormlab.environments.gym_env import DoomwormEnv
from wormlab.environments.sensors import PRESETS


class _Capture:
    """Inner 'brain' of the planner layer that just records the merged channels."""

    name = "capture"

    def __init__(self) -> None:
        self.channels: dict[str, float] = {}

    def reset(self) -> None:
        """Forget the last frame."""
        self.channels = {}

    def act(self, channels: Mapping[str, float]) -> tuple[float, float]:
        """Record and stand still; the policy, not this, drives."""
        self.channels = dict(channels)
        return 0.0, 0.0


class PlannerWrapper(gym.Wrapper[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
    """The map + planner + needs layer (Plan §3.2) as a Gymnasium wrapper."""

    def __init__(self, env: DoomwormEnv, mode: str = "needs") -> None:
        super().__init__(env)
        self.base = env
        self.capture = _Capture()
        self.layer = PlannerLayer(self.capture, PRESETS[env.sensor_preset], mode=mode)

    def _merged(self, channels: dict[str, float]) -> np.ndarray:
        self.layer.act(channels)
        merged = self.capture.channels or (dict(channels) | self.layer.gradient())
        return self.base.vector(merged)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Fresh map and arbiter with the fresh world."""
        _, info = self.base.reset(seed=seed, options=options)
        self.layer.reset()
        self.capture.reset()
        return self._merged(self.base.last_channels), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Parked on the dock: wheels stopped, whatever the policy says."""
        if self.layer.parked():
            action = np.zeros(len(action))
        _, reward, terminated, truncated, info = self.base.step(action)
        return self._merged(self.base.last_channels), reward, terminated, truncated, info
