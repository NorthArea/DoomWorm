"""Units and the raw-reading contract between the machine and the brain (Plan §2.4, §20.5).

The simulator speaks abstract world units and normalised channels in [0, 1].
The robot speaks metres, radians and volts. This module is the only place the
two meet: a :class:`RawReading` is what the firmware sends (physical units), a
:class:`Calibration` turns it into exactly the channels the vacuum
:class:`~doomworm.environments.sensors.SensorSuite` emits, in the same order.

Scale assumptions (docs/hardware.md, "assumed, not measured"): ``unit_m`` metres
per world unit and ``tick_s`` seconds per environment step are adjusted once
the hardware is chosen; nothing in the brain depends on them.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from doomworm.environments.sensors import PASSTHROUGH, VACUUM, SensorConfig


@dataclass(frozen=True)
class RawReading:
    """One sensor frame from the machine, in physical units.

    ``ranges_m``: distance per ray, in the suite's ray order (left .. right);
    ``None`` = no echo (nothing within range). ``bumper``/``cliff``: (left, right)
    flags. ``wall_m``: right-side IR distance or ``None``. ``odom_m``/``odom_rad``:
    dead-reckoning pose from wheel encoders, ``gyro_rad``: heading from the IMU.
    ``battery``: state of charge in [0, 1]; ``charging``: on the dock contacts.
    ``dock``: IR beacon strength per sector (left, front, right) in [0, 1].
    """

    ranges_m: list[float | None] = field(default_factory=list)
    bumper: tuple[int, int] = (0, 0)
    cliff: tuple[int, int] = (0, 0)
    wall_m: float | None = None
    odom_m: tuple[float, float] = (0.0, 0.0)
    odom_rad: float = 0.0
    gyro_rad: float = 0.0
    battery: float = 1.0
    charging: int = 0
    dock: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tick: int = 0

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form (the wire message)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RawReading:
        """Parse a wire message; unknown keys are ignored, missing ones take defaults."""
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        for key in ("bumper", "cliff", "odom_m", "dock"):
            if key in known:
                known[key] = tuple(known[key])
        return cls(**known)


@dataclass(frozen=True)
class Calibration:
    """Metres and seconds of the machine against world units and ticks."""

    unit_m: float = 0.33  # metres per world unit: body radius 0.5 u = 0.165 m, a 33 cm vacuum
    tick_s: float = 0.1  # seconds per environment step
    sensors: SensorConfig = VACUUM

    # --- derived physical ranges ----------------------------------------------------

    @property
    def ray_range_m(self) -> float:
        """Rangefinder reach in metres (suite ``ray_range`` scaled)."""
        return self.sensors.ray_range * self.unit_m

    @property
    def wall_range_m(self) -> float:
        """Side IR reach in metres."""
        return self.sensors.wall_range * self.unit_m

    @property
    def cliff_reach_m(self) -> float:
        """How far ahead of the centre the cliff sensors look, in metres."""
        return self.sensors.cliff_reach * self.unit_m

    def to_units(self, metres: float) -> float:
        """Metres -> world units."""
        return metres / self.unit_m

    def to_metres(self, units: float) -> float:
        """World units -> metres."""
        return units * self.unit_m

    def proximity(self, metres: float | None, range_m: float) -> float:
        """Distance -> the suite's proximity ``1 - d / range`` in [0, 1]; no echo = 0."""
        if metres is None:
            return 0.0
        return max(0.0, min(1.0, 1.0 - metres / range_m))

    # --- raw -> channels -------------------------------------------------------------

    def channels(self, raw: RawReading) -> dict[str, float]:
        """The brain's channel dict, keys and order identical to the simulator's suite."""
        cfg = self.sensors
        n = len(cfg.ray_angles)
        if len(raw.ranges_m) != n:
            raise ValueError(f"{cfg.name} preset has {n} rays, reading has {len(raw.ranges_m)}")
        out: dict[str, float] = {}
        rays = []
        for i, (angle, metres) in enumerate(zip(cfg.ray_angles, raw.ranges_m, strict=True)):
            p = self.proximity(metres, self.ray_range_m)
            out[f"range_{i}"] = p
            rays.append((angle, p))
        fa = cfg.front_half_angle
        out["sensor_left"] = max((p for a, p in rays if a > fa), default=0.0)
        out["sensor_front"] = max((p for a, p in rays if -fa <= a <= fa), default=0.0)
        out["sensor_right"] = max((p for a, p in rays if a < -fa), default=0.0)
        if cfg.bumper:
            out["bumper_left"], out["bumper_right"] = float(raw.bumper[0]), float(raw.bumper[1])
        if cfg.cliff:
            out["cliff_left"], out["cliff_right"] = float(raw.cliff[0]), float(raw.cliff[1])
        if cfg.wall_sensor:
            out["wall_right"] = self.proximity(raw.wall_m, self.wall_range_m)
        if cfg.odometry:
            out["odom_x"] = self.to_units(raw.odom_m[0])
            out["odom_y"] = self.to_units(raw.odom_m[1])
            out["odom_heading"] = raw.odom_rad
            out["gyro_heading"] = raw.gyro_rad
        battery = max(0.0, min(1.0, raw.battery))
        beacon = {"dock_left": raw.dock[0], "dock_front": raw.dock[1], "dock_right": raw.dock[2]}
        for key in PASSTHROUGH:
            if key == "battery":
                out[key] = battery
            elif key == "hunger":
                out[key] = 1.0 - battery
            elif key == "health":
                out[key] = 1.0
            else:
                out[key] = float(beacon.get(key, 0.0))
        return out

    # --- channels -> raw (the simulator pretending to be a machine) ----------------------

    def raw_from_channels(self, channels: dict[str, float], tick: int = 0) -> RawReading:
        """Inverse of :meth:`channels`; ``proximity 0`` becomes "no echo" (``None``)."""
        cfg = self.sensors

        def metres(p: float, range_m: float) -> float | None:
            return None if p <= 0.0 else (1.0 - p) * range_m

        ranges = [
            metres(channels[f"range_{i}"], self.ray_range_m) for i in range(len(cfg.ray_angles))
        ]
        return RawReading(
            ranges_m=ranges,
            bumper=(int(channels.get("bumper_left", 0.0)), int(channels.get("bumper_right", 0.0))),
            cliff=(int(channels.get("cliff_left", 0.0)), int(channels.get("cliff_right", 0.0))),
            wall_m=metres(channels.get("wall_right", 0.0), self.wall_range_m),
            odom_m=(
                self.to_metres(channels.get("odom_x", 0.0)),
                self.to_metres(channels.get("odom_y", 0.0)),
            ),
            odom_rad=channels.get("odom_heading", 0.0),
            gyro_rad=channels.get("gyro_heading", 0.0),
            battery=channels.get("battery", 1.0),
            charging=0,
            dock=(
                channels.get("dock_left", 0.0),
                channels.get("dock_front", 0.0),
                channels.get("dock_right", 0.0),
            ),
            tick=tick,
        )

    def to_dict(self) -> dict[str, Any]:
        """Meta for logs."""
        return {
            "unit_m": self.unit_m,
            "tick_s": self.tick_s,
            "sensors": self.sensors.name,
            "ray_angles_deg": [round(math.degrees(a), 1) for a in self.sensors.ray_angles],
            "ray_range_m": self.ray_range_m,
            "wall_range_m": self.wall_range_m,
        }
