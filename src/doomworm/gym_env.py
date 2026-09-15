"""Gymnasium wrapper around the 2D world (Plan §3.3, stage 18).

Observation: a fixed-order vector of sensor channels. Action: two wheel
commands in [-1, 1]. Episodes end when the agent is dead (terminated) or the
step cap is hit (truncated). Everything is seeded through ``reset(seed=...)``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import gymnasium as gym
import numpy as np

from doomworm.brains.planner_layer import PlannerLayer
from doomworm.environments.sensors import PASSTHROUGH, PRESETS, SensorSuite
from doomworm.environments.simple_2d import Observation, World
from doomworm.learning.reward import RewardConfig, RewardTracker

WorldFactory = Callable[[int], World]


class DoomwormEnv(gym.Env[np.ndarray, np.ndarray]):
    """One agent, wheels in, channels out."""

    metadata = {"render_modes": ["ansi"]}  # noqa: RUF012 - gymnasium declares it this way

    def __init__(
        self,
        world_factory: WorldFactory,
        sensors: str = "ideal",
        steps: int = 800,
        reward_config: RewardConfig | None = None,
        channel_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.world_factory = world_factory
        self.sensor_preset = sensors
        self.steps = steps
        self.reward_config = reward_config or RewardConfig()
        probe = SensorSuite(PRESETS[sensors])
        self.channel_names = list(channel_names or probe.channel_names)
        n = len(self.channel_names)
        unbounded = {"odom_x", "odom_y", "odom_heading", "gyro_heading"}
        low = np.array([-np.inf if c in unbounded else 0.0 for c in self.channel_names])
        high = np.array([np.inf if c in unbounded else 1.0 for c in self.channel_names])
        self.observation_space = gym.spaces.Box(low, high, shape=(n,), dtype=np.float64)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float64)
        self.world: World | None = None
        self.suite: SensorSuite | None = None
        self.tracker = RewardTracker(self.reward_config)
        self.tick = 0
        self._obs: Observation | None = None
        self._seed = 0
        self.last_channels: dict[str, float] = {}

    def _vector(self, channels: dict[str, float]) -> np.ndarray:
        return np.array([channels.get(c, 0.0) for c in self.channel_names], dtype=np.float64)

    def _channels(self) -> dict[str, float]:
        assert self.world is not None
        assert self._obs is not None
        if self.suite is None:
            return self._obs.as_channels()
        return self.suite.read(self.world, self._obs)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Build a fresh seeded world (and sensor suite) and return the first observation."""
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
        self.world = self.world_factory(self._seed)
        # Every preset goes through the suite: ideal is noiseless, not sensorless.
        self.suite = SensorSuite(PRESETS[self.sensor_preset], seed=self._seed)
        self.suite.reset(self.world)
        self.tracker = RewardTracker(self.reward_config)
        self.tick = 0
        self._obs = self.world.observe()
        self.last_channels = self._channels()
        return self._vector(self.last_channels), {"seed": self._seed}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply wheel commands for one environment step."""
        assert self.world is not None, "call reset() first"
        left, right = float(action[0]), float(action[1])
        obs = self.world.step(left, right)
        self._obs = obs
        self.tick += 1
        reward = self.tracker.step(
            x=self.world.agent.x,
            y=self.world.agent.y,
            ate=obs.ate,
            collided=obs.collided,
            starved=self.world.starved,
            reached=obs.reached,
            damaged=obs.damaged,
            dead=self.world.dead,
            cleaned=obs.cleaned,
            docked=obs.docked,
            battery=obs.battery,
        )
        terminated = self.world.dead
        truncated = self.tick >= self.steps and not terminated
        info = {
            "tick": self.tick,
            "coverage": self.world.coverage,
            "food": self.world.food_eaten,
            "targets": self.world.targets_reached,
            "collisions": self.world.collisions,
            "damage": self.world.damage_taken,
            "dockings": self.world.dockings,
            "total_reward": self.tracker.total,
        }
        self.last_channels = self._channels()
        return self._vector(self.last_channels), reward, terminated, truncated, info

    def render(self) -> str:
        """ASCII snapshot."""
        from doomworm.episode import Record
        from doomworm.experiments.episode import render_ascii

        assert self.world is not None
        a = self.world.agent
        rec = Record(0, a.x, a.y, a.heading, (0, 0, 0), (0, 0, 0), 0.0, (0.0, 0.0), False, False)
        return render_ascii(self.world, [rec])


class _Capture:
    """Inner 'brain' of the planner layer that just records the merged channels."""

    name = "capture"

    def __init__(self) -> None:
        self.channels: dict[str, float] = {}

    def reset(self) -> None:
        self.channels = {}

    def act(self, channels: Mapping[str, float]) -> tuple[float, float]:
        self.channels = dict(channels)
        return 0.0, 0.0


class PlannerWrapper(gym.Wrapper[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
    """The map + planner + needs layer (Plan §3.2) as a Gymnasium wrapper.

    The policy sees the same channels a wrapped Brain would: the planner's
    virtual ``target_*`` gradient replaces the raw one, and while the robot
    sits on the dock charging the action is overridden with stopped wheels.
    So an RL policy is compared under exactly the layer the other candidates get.
    """

    def __init__(self, env: DoomwormEnv, mode: str = "needs") -> None:
        super().__init__(env)
        self.base = env
        self.capture = _Capture()
        self.layer = PlannerLayer(self.capture, PRESETS[env.sensor_preset], mode=mode)

    def _merged(self, channels: dict[str, float]) -> np.ndarray:
        self.layer.act(channels)
        merged = self.capture.channels or (dict(channels) | self.layer.gradient())
        return self.base._vector(merged)

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
            action = np.zeros(2)
        _, reward, terminated, truncated, info = self.base.step(action)
        return self._merged(self.base.last_channels), reward, terminated, truncated, info


def passthrough_names() -> list[str]:
    """Names of the non-ranging channels (beacons and internal state)."""
    return list(PASSTHROUGH)
