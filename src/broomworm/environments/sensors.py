"""Emulated sensor suite of a home vacuum (Plan §20.3, stage 17).

A :class:`SensorSuite` sits between the world and the brain. It turns the
ideal :class:`~broomworm.environments.simple_2d.Observation` plus world state
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

from broomworm.environments.simple_2d import Observation, World


@dataclass(frozen=True)
class SensorConfig:
    """Shape and imperfections of the suite; presets below."""

    name: str = "ideal"
    body: str = "differential"  # the vehicle the platform drives (Plan §20.5.1)
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
    # --- stage 22: what a kit car has instead of vacuum sensors ---
    sweep: bool = False  # one rangefinder on a servo: ray i is refreshed every len(rays) ticks
    proximity_bumper: float = 0.0  # no bumper: a front ray closer than this (units) reads as a hit
    wall_binary: float = 0.0  # IR obstacle module on the side: 1 within this distance, else 0
    odom_source: str = "wheels"  # "wheels" = encoders, "commands" = open loop, blind to slip
    gyro: bool = True  # False: no IMU, gyro_heading is the odometry heading
    beacon_fov: float | None = None  # camera marker as the dock beacon: half-angle of view
    beacon_range: float | None = None  # ... and how far the marker is recognised


# ideal = every sensor of the vacuum, three 45-degree rays, no noise, no delay
IDEAL = SensorConfig(bumper=True, cliff=True, wall_sensor=True, odometry=True)
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
# A kit car (stage 22: ACEBOTT QD001 + QD003 with one or two spare sensors): one
# ultrasonic on a servo swept over three headings, an IR obstacle module on the
# right, line-tracking modules under the nose as cliff sensors, no bumper, no
# encoders, no IMU; the K210 camera recognises the dock marker within its view.
CAR = SensorConfig(
    body="mecanum",  # QD001: four mecanum wheels, so it can also move sideways
    name="car",
    ray_angles=tuple(math.radians(a) for a in (30.0, 0.0, -30.0)),
    ray_range=4.0,
    noise_sigma=0.05,
    dropout=0.05,
    delay=1,
    sweep=True,
    proximity_bumper=0.7,
    cliff=True,
    wall_sensor=True,
    wall_range=1.0,
    wall_binary=0.75,
    odometry=True,
    odom_sigma=0.1,
    odom_source="commands",
    gyro=False,
    beacon_fov=math.radians(30.0),
    beacon_range=5.0,
)
PRESETS: dict[str, SensorConfig] = {"ideal": IDEAL, "vacuum": VACUUM, "noisy": NOISY, "car": CAR}

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
    _held: list[float] = field(init=False, default_factory=list)  # sweep: last value per ray
    _tick: int = field(init=False, default=0)

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
        if cfg.bumper or cfg.proximity_bumper:
            names += ["bumper_left", "bumper_right"]
        if cfg.cliff:
            names += ["cliff_left", "cliff_right"]
        if cfg.wall_sensor:
            names.append("wall_right")
        if cfg.odometry:
            names += ["odom_x", "odom_y", "odom_heading", "gyro_heading"]
        return names + list(PASSTHROUGH)

    def reset(self, world: World) -> None:
        """Start an episode: odometry begins at the true pose, sensors already warm.

        The delay buffer is pre-filled with the reading at rest, so the first
        ``delay`` ticks report stale values, not zeros: a robot that has been sitting
        on its dock knows where it is and what is in front of it.
        """
        self._buffer.clear()
        a = world.agent
        self.odom = [a.x, a.y, a.heading]
        self._last_pose = (a.x, a.y, a.heading)
        self.gyro_bias = 0.0
        self._tick = 0
        if self.config.sweep:  # a full sweep at rest; other presets keep their RNG stream
            self._held = [self._ray(world, angle) for angle in self.config.ray_angles]
        if self.config.delay:
            warm = self._measure(world, world.observe())
            for _ in range(self.config.delay):
                self._buffer.append(dict(warm))

    def read(self, world: World, obs: Observation) -> dict[str, float]:
        """Channels for the brain this tick (delayed by ``config.delay`` steps)."""
        fresh = self._measure(world, obs)
        self._buffer.append(fresh)
        if len(self._buffer) <= self.config.delay:
            return dict.fromkeys(fresh, 0.0) | {k: fresh[k] for k in PASSTHROUGH}
        return self._buffer[0]

    # --- measurements ------------------------------------------------------------

    def _ray(self, world: World, angle: float) -> float:
        """One noisy proximity reading along ``heading + angle``."""
        cfg = self.config
        d = world.ray_distance(world.agent.heading + angle)
        p = max(0.0, min(1.0, 1.0 - d / cfg.ray_range))
        if cfg.noise_sigma:
            p = max(0.0, min(1.0, p * (1.0 + self.rng.normal(0.0, cfg.noise_sigma))))
        if cfg.dropout and self.rng.random() < cfg.dropout:
            p = 0.0
        return p

    def _measure(self, world: World, obs: Observation) -> dict[str, float]:
        cfg = self.config
        out: dict[str, float] = {}
        rays = []
        n = len(cfg.ray_angles)
        for i, angle in enumerate(cfg.ray_angles):
            if cfg.sweep and self._tick % n != i:
                p = self._held[i]  # the servo is pointing elsewhere this tick
            else:
                p = self._ray(world, angle)
                if cfg.sweep:
                    self._held[i] = p
            out[f"range_{i}"] = p
            rays.append((angle, p))
        self._tick += 1
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
        elif cfg.proximity_bumper:
            # No bumper: a ray that reads closer than the threshold counts as contact on
            # its side (the front ray on both). Fires before touching, misses what the
            # rays do not see (low or side obstacles), never knows about a real collision.
            near = 1.0 - cfg.proximity_bumper / cfg.ray_range
            left = float(any(p > near for a, p in rays if a >= 0.0))
            right = float(any(p > near for a, p in rays if a <= 0.0))
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
            if cfg.wall_binary:
                out["wall_right"] = 1.0 if d < cfg.wall_binary else 0.0
            else:
                out["wall_right"] = max(0.0, min(1.0, 1.0 - d / cfg.wall_range))

        if cfg.odometry:
            self._integrate_odometry(world)
            out["odom_x"], out["odom_y"], out["odom_heading"] = self.odom
            if cfg.gyro:
                self.gyro_bias += self.rng.normal(0.0, cfg.gyro_drift) if cfg.gyro_drift else 0.0
                out["gyro_heading"] = world.agent.heading + self.gyro_bias
            else:
                out["gyro_heading"] = self.odom[2]

        channels = obs.as_channels()
        for key in PASSTHROUGH:
            out[key] = channels.get(key, 0.0)
        if cfg.beacon_fov is not None:
            out["dock_left"], out["dock_front"], out["dock_right"] = self._camera_beacon(world)
        return out

    def _camera_beacon(self, world: World) -> tuple[float, float, float]:
        """The dock marker as a camera sees it: only inside the field of view and range."""
        cfg = self.config
        dock = world.dock
        if dock is None or cfg.beacon_fov is None:
            return 0.0, 0.0, 0.0
        dx, dy = dock.x - world.agent.x, dock.y - world.agent.y
        distance = math.hypot(dx, dy)
        if cfg.beacon_range is not None and distance > cfg.beacon_range:
            return 0.0, 0.0, 0.0
        bearing = math.remainder(math.atan2(dy, dx) - world.agent.heading, math.tau)
        if abs(bearing) > cfg.beacon_fov:
            return 0.0, 0.0, 0.0
        if world.ray_distance(math.atan2(dy, dx)) < distance - dock.radius:
            return 0.0, 0.0, 0.0  # a wall or furniture in the way: a camera cannot see through
        signal = 1.0 if distance <= 1.0 else 1.0 / distance
        if abs(bearing) <= cfg.beacon_fov / 3.0:
            return 0.0, signal, 0.0
        return (signal, 0.0, 0.0) if bearing > 0.0 else (0.0, 0.0, signal)

    def _integrate_odometry(self, world: World) -> None:
        a = world.agent
        if self.config.odom_source == "commands":
            # No encoders: integrate what the motors were told. A robot pushing against
            # a wall or slipping on a rug keeps "moving" on this odometry.
            left, right = world.last_command
            dist = (left + right) / 2.0 * world.speed
            dh = (right - left) / world.wheel_base * world.speed
            if self.config.odom_sigma:
                dist *= 1.0 + self.rng.normal(0.0, self.config.odom_sigma)
                dh *= 1.0 + self.rng.normal(0.0, self.config.odom_sigma)
            self.odom[2] += dh
            self.odom[0] += dist * math.cos(self.odom[2])
            self.odom[1] += dist * math.sin(self.odom[2])
            self._last_pose = (a.x, a.y, a.heading)
            return
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
