"""Day-one checks of a machine behind a link (stage 22.2): protocol, sensors, wheels.

``doomworm selftest`` connects, reads frames at rest, wiggles the wheels and
reports what moved. It is the first thing to run on the real car and the
last thing to run before a drive; it works on the simulator link too, which is
how it is tested.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field

from doomworm.hardware.link import RobotLink


@dataclass
class SelfTestReport:
    """What the self-test saw."""

    channels: dict[str, float]
    rest_frames: int
    roundtrip_ms: list[float] = field(default_factory=list)
    dropouts: dict[str, int] = field(default_factory=dict)
    heading_left_spin: float = 0.0  # odometry heading change when only the right wheel drives
    heading_right_spin: float = 0.0
    forward_units: float = 0.0
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # true, but not a failure

    @property
    def ok(self) -> bool:
        """No problem found."""
        return not self.problems

    def markdown(self) -> str:
        """Report."""
        lines = [f"frames at rest: {self.rest_frames}"]
        if self.roundtrip_ms:
            lines.append(
                f"round trip: mean {statistics.fmean(self.roundtrip_ms):.1f} ms, "
                f"max {max(self.roundtrip_ms):.1f} ms"
            )
        lines.append(
            f"spin: right wheel only -> heading {self.heading_left_spin:+.3f} rad, "
            f"left wheel only -> {self.heading_right_spin:+.3f} rad; "
            f"forward 5 ticks -> {self.forward_units:.3f} u"
        )
        lines += ["", "| channel | at rest | dropouts |", "|---|---|---|"]
        for k, v in self.channels.items():
            lines.append(f"| {k} | {v:.3f} | {self.dropouts.get(k, 0)} |")
        lines.append("")
        lines += [f"NOTE: {n}" for n in self.notes]
        lines += [f"PROBLEM: {p}" for p in self.problems] or ["OK"]
        return "\n".join(lines) + "\n"


def _sensor_config(link: RobotLink) -> str | None:
    """The link's odometry source: ``wheels`` (encoders), ``commands`` (open loop)."""
    for owner, attr in ((link, "suite"), (link, "calibration")):
        holder = getattr(owner, attr, None)
        cfg = getattr(holder, "config", None) or getattr(holder, "sensors", None)
        if cfg is not None:
            return str(getattr(cfg, "odom_source", "wheels"))
    return None


def _range_keys(channels: dict[str, float]) -> list[str]:
    return [k for k in channels if k.startswith("range_")]


def selftest(link: RobotLink, rest_frames: int = 10, expect_move: bool = True) -> SelfTestReport:
    """Reset, watch the sensors at rest, wiggle each side, drive forward, stop."""
    first = link.reset()
    report = SelfTestReport(channels=dict(first), rest_frames=rest_frames)
    expected = set(link.channel_names)
    missing = expected - set(first)
    if missing:
        report.problems.append(f"channels missing from the frame: {sorted(missing)}")
    for key, value in first.items():
        if not math.isfinite(value):
            report.problems.append(f"{key} is not a number at rest")
    if not 0.0 <= first.get("battery", 1.0) <= 1.0:
        report.problems.append("battery outside [0, 1]")

    # --- at rest: latency and dropouts ---------------------------------------------
    dropouts = dict.fromkeys(_range_keys(first), 0)
    for _ in range(rest_frames):
        t0 = time.perf_counter()
        frame = link.step(0.0, 0.0)
        report.roundtrip_ms.append((time.perf_counter() - t0) * 1000.0)
        for key in dropouts:
            if frame.get(key, 0.0) == 0.0 and first.get(key, 0.0) > 0.0:
                dropouts[key] += 1
    report.dropouts = dropouts
    for key, n in dropouts.items():
        if n > rest_frames // 2:
            report.problems.append(f"{key} reads nothing in {n}/{rest_frames} frames at rest")

    # --- wiggle: does odometry answer the wheels the right way round ----------------
    dead_reckoned = _sensor_config(link) is not None and _sensor_config(link) == "commands"
    frame = link.step(0.0, 0.0)
    h0 = frame.get("odom_heading", 0.0)
    for _ in range(5):
        frame = link.step(0.0, 1.0)  # right wheel only: turn left, heading up
    report.heading_left_spin = frame.get("odom_heading", 0.0) - h0
    for _ in range(5):
        frame = link.step(1.0, 0.0)
    report.heading_right_spin = frame.get("odom_heading", 0.0) - h0 - report.heading_left_spin
    x0, y0 = frame.get("odom_x", 0.0), frame.get("odom_y", 0.0)
    for _ in range(5):
        frame = link.step(1.0, 1.0)
    report.forward_units = math.dist((frame.get("odom_x", 0.0), frame.get("odom_y", 0.0)), (x0, y0))
    link.step(0.0, 0.0)
    if dead_reckoned:
        # No encoders: the odometry below is integrated from the commands the host
        # itself sent, so these three checks test the host's arithmetic, not the car.
        report.notes.append(
            "odometry is dead reckoning on this preset (no encoders): the spin and "
            "forward checks test the host, not the wiring. Confirm by eye that each "
            "wheel pair turns the way the command says before trusting a drive."
        )
    if expect_move and "odom_heading" in first:
        if report.heading_left_spin <= 0.0:
            report.problems.append("right wheel alone did not turn the heading left (+)")
        if report.heading_right_spin >= 0.0:
            report.problems.append("left wheel alone did not turn the heading right (-)")
        if report.forward_units <= 0.0:
            report.problems.append("forward command did not advance the odometry")
    return report
