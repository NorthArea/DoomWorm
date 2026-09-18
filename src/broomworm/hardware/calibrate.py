"""Measure the scale of the machine (stage 22.2): metres per unit, wheel base, tick.

``broomworm calibrate`` drives forward for N ticks and asks how far the car
went, then spins for N ticks and asks how far it turned. From the two
answers it derives the numbers `Calibration` assumes today (unit_m, speed,
wheel_base) and writes them to a JSON file that ``--calibration`` loads.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from broomworm.hardware.calibration import Calibration
from broomworm.hardware.link import RobotLink


def _drive(link: RobotLink, left: float, right: float, ticks: int) -> float:
    t0 = time.perf_counter()
    for _ in range(ticks):
        link.step(left, right)
    link.step(0.0, 0.0)
    return (time.perf_counter() - t0) / max(ticks, 1)


def calibrate(
    link: RobotLink,
    base: Calibration,
    ticks: int = 20,
    ask: Callable[[str], str] | None = None,
    say: Callable[[str], None] = print,
) -> Calibration:
    """Forward run + spin, two questions, a calibration with measured numbers."""
    ask = ask if ask is not None else input  # resolved at call time (tests patch input)
    link.reset()
    say(f"forward at full speed for {ticks} ticks; measure the straight-line distance")
    tick_s = _drive(link, 1.0, 1.0, ticks)
    metres = float(ask("distance travelled, metres: "))
    units = ticks * base.speed  # what the simulator would have driven
    unit_m = metres / units if units else base.unit_m

    say(f"spinning in place for {ticks} ticks; measure the angle turned")
    _drive(link, -1.0, 1.0, ticks)
    degrees = float(ask("angle turned, degrees (positive = left / counter-clockwise): "))
    radians = math.radians(abs(degrees))
    # in the simulator a spin of N ticks turns 2 * speed * N / wheel_base radians
    wheel_base = (2.0 * base.speed * ticks) / radians if radians else base.wheel_base

    measured = replace(base, unit_m=unit_m, wheel_base=wheel_base, tick_s=tick_s)
    say(
        f"measured: unit_m {unit_m:.3f} (was {base.unit_m}), wheel_base {wheel_base:.2f} u "
        f"(was {base.wheel_base}), tick {tick_s * 1000:.0f} ms, "
        f"top speed {measured.speed_mps:.2f} m/s"
    )
    return measured


def save_calibration(cal: Calibration, path: Path | str) -> Path:
    """Write the measured numbers (the preset is named, not copied)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "sensors": cal.sensors.name,
        "unit_m": cal.unit_m,
        "tick_s": cal.tick_s,
        "speed": cal.speed,
        "wheel_base": cal.wheel_base,
    }
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out


def load_calibration(path: Path | str) -> Calibration:
    """Read a file written by :func:`save_calibration`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    base = Calibration.for_preset(str(data.get("sensors", "car")))
    return replace(
        base,
        unit_m=float(data.get("unit_m", base.unit_m)),
        tick_s=float(data.get("tick_s", base.tick_s)),
        speed=float(data.get("speed", base.speed)),
        wheel_base=float(data.get("wheel_base", base.wheel_base)),
    )
