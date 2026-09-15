"""Emulated sensor suite of a home vacuum (Plan §20.3, stage 17).

A :class:`SensorSuite` sits between the world and the brain. It turns the
ideal :class:`~doomworm.environments.simple_2d.Observation` plus world state
into the channels a real robot would have, with noise, dropouts and delay:

    range_<i>                 proximity per ray, 1 - d / range in [0, 1]
    sensor_left/front/right   legacy channels: strongest ray per sector, so
                              every existing mapping keeps working
    bumper_left/right         contact this tick, side by nearest front ray
    cliff_left/right          a hazard edge just ahead of the left/right wheel
    wall_right                side IR: 1 - d / wall_range along -90 degrees
    odom_x/odom_y/odom_heading  dead reckoning with multiplicative noise + drift
    gyro_heading              heading with a slow random-walk bias

Gradient channels (food_*, target_*, danger_*, dock_*, hunger, battery,
health) pass through unchanged: they are beacons or internal state, not
ranging sensors. All randomness comes from the suite's own seeded RNG.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from doomworm.environments.simple_2d import Observation, World


@dataclass(frozen=True)
class SensorConfig:
    """Shape and imperfections of the suite; presets below."""

    name: str = "ideal"
    ray_angles: tuple[float, ...] = (math.pi / 4, 0.0, -math.pi / 4)  # left .. right
    ray_range: float = 4.0
    noise_sigma: float = 0.0  # relative gaussian noise on proximity readings
    dropout: float = 0.0  # probability a ray reads 0 this tick
    delay: int = 0  # environment steps of latency
    bumper: bool = False
    cliff: bool = False
    cliff_reach: float = 0.7  # distance ahead of the centre that the cliff sensors probe
    wall_sensor: bool = False
    wall_range: float = 1.0
    odometry: bool = False
    odom_sigma: float = 0.0  # relative noise on each displacement / rotation
    gyro_drift: float = 0.0  # random-walk step (rad) of the heading bias per tick
    front_half_angle: float = math.radians(15.0)


IDEAL = SensorConfig()
VACUUM = SensorConfig(
    name="vacuum",
    ray_angles=tuple(math.radians(a) for a in (60.0, 30.0, 0.0, -30.0, -60.0)),
    noise_sigma=0.05,
    dropout=0.02,
    delay=1,
    bumper=True,
    cliff=True,
    wall_sensor=True,
    odometry=True,
    odom_sigma=0.05,
    gyro_drift=0.002,
)
NOISY = SensorConfig(
    name="noisy",
    ray_angles=VACUUM.ray_angles,
    noise_sigma=0.15,
    dropout=0.05,
    delay=2,
    bumper=True,
    cliff=True,
    wall_sensor=True,
    odometry=True,
    odom_sigma=0.15,
    gyro_drift=0.01,
)
PRESETS: dict[str, SensorConfig] = {"ideal": IDEAL, "vacuum": VACUUM, "noisy": NOISY}

PASSTHROUGH = (
    "food_left", "food_front", "food_right", "hunger",
    "target_left", "target_front", "target_right",
    "danger_left", "danger_front", "danger_right", "health",
    "battery", "dock_left", "dock_front", "dock_right",
)  # fmt: skip


@dataclass
class SensorSuite:
    """Stateful sensor emulation for one episode."""

    config: SensorConfig = IDEAL
    seed: int = 0
    rng: np.random.Generator = field(init=False)
    _buffer: deque[dict[str, float]] = field(init=False)
    _last_pose: tuple[float, float, float] | None = field(init=False, default=None)
    odom: list[float] = field(init=False)
    gyro_bias: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self._buffer = deque(maxlen=self.config.delay + 1)
        self.odom = [0.0, 0.0, 0.0]

    @property
    def channel_names(self) -> list[str]:
        """Every channel this suite emits, in a fixed order."""
        cfg = self.config
        names = [f"range_{i}" for i in range(len(cfg.ray_angles))]
        names += ["sensor_left", "sensor_front", "sensor_right"]
        if cfg.bumper:
            names += ["bumper_left", "bumper_right"]
        if cfg.cliff:
            names += ["cliff_left", "cliff_right"]
        if cfg.wall_sensor:
            names.append("wall_right")
        if cfg.odometry:
            names += ["odom_x", "odom_y", "odom_heading", "gyro_heading"]
        return names + list(PASSTHROUGH)

    def reset(self, world: World) -> None:
        """Start an episode: odometry begins at the true pose, buffers empty."""
        self._buffer.clear()
        a = world.agent
        self.odom = [a.x, a.y, a.heading]
        self._last_pose = (a.x, a.y, a.heading)
        self.gyro_bias = 0.0

    def read(self, world: World, obs: Observation) -> dict[str, float]:
        """Channels for the brain this tick (delayed by ``config.delay`` steps)."""
        fresh = self._measure(world, obs)
        self._buffer.append(fresh)
        if len(self._buffer) <= self.config.delay:
            return dict.fromkeys(fresh, 0.0) | {k: fresh[k] for k in PASSTHROUGH}
        return self._buffer[0]

    # --- measurements ------------------------------------------------------------

    def _measure(self, world: World, obs: Observation) -> dict[str, float]:
        cfg = self.config
        out: dict[str, float] = {}
        rays = []
        for i, angle in enumerate(cfg.ray_angles):
            d = world.ray_distance(world.agent.heading + angle)
            p = max(0.0, min(1.0, 1.0 - d / cfg.ray_range))
            if cfg.noise_sigma:
                p = max(0.0, min(1.0, p * (1.0 + self.rng.normal(0.0, cfg.noise_sigma))))
            if cfg.dropout and self.rng.random() < cfg.dropout:
                p = 0.0
            out[f"range_{i}"] = p
            rays.append((angle, p))
        fa = cfg.front_half_angle
        out["sensor_left"] = max((p for a, p in rays if a > fa), default=0.0)
        out["sensor_front"] = max((p for a, p in rays if -fa <= a <= fa), default=0.0)
        out["sensor_right"] = max((p for a, p in rays if a < -fa), default=0.0)

        if cfg.bumper:
            left = right = 0.0
            if obs.collided:
                lp = max((p for a, p in rays if a > 0), default=0.0)
                rp = max((p for a, p in rays if a < 0), default=0.0)
                left, right = (1.0, 0.0) if lp > rp else (0.0, 1.0) if rp > lp else (1.0, 1.0)
            out["bumper_left"], out["bumper_right"] = left, right

        if cfg.cliff:
            for name, side in (("cliff_left", 1.0), ("cliff_right", -1.0)):
                ang = world.agent.heading + side * math.radians(30.0)
                px = world.agent.x + math.cos(ang) * cfg.cliff_reach
                py = world.agent.y + math.sin(ang) * cfg.cliff_reach
                inside = any(math.dist((px, py), (d.x, d.y)) < d.radius for d in world.dangers)
                out[name] = 1.0 if inside else 0.0

        if cfg.wall_sensor:
            d = world.ray_distance(world.agent.heading - math.pi / 2)
            out["wall_right"] = max(0.0, min(1.0, 1.0 - d / cfg.wall_range))

        if cfg.odometry:
            self._integrate_odometry(world)
            out["odom_x"], out["odom_y"], out["odom_heading"] = self.odom
            self.gyro_bias += self.rng.normal(0.0, cfg.gyro_drift) if cfg.gyro_drift else 0.0
            out["gyro_heading"] = world.agent.heading + self.gyro_bias

        channels = obs.as_channels()
        for key in PASSTHROUGH:
            out[key] = channels.get(key, 0.0)
        return out

    def _integrate_odometry(self, world: World) -> None:
        a = world.agent
        if self._last_pose is None:
            self.odom = [a.x, a.y, a.heading]
        else:
            lx, ly, lh = self._last_pose
            dist = math.dist((a.x, a.y), (lx, ly))
            dh = a.heading - lh
            forward = math.cos(a.heading - math.atan2(a.y - ly, a.x - lx)) >= 0 if dist else True
            if self.config.odom_sigma:
                dist *= 1.0 + self.rng.normal(0.0, self.config.odom_sigma)
                dh *= 1.0 + self.rng.normal(0.0, self.config.odom_sigma)
            self.odom[2] += dh
            sign = 1.0 if forward else -1.0
            self.odom[0] += sign * dist * math.cos(self.odom[2])
            self.odom[1] += sign * dist * math.sin(self.odom[2])
        self._last_pose = (a.x, a.y, a.heading)
