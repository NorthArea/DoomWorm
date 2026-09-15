"""Sim-vs-real check (Plan §20.5, the comparison against the simulator).

Replay the wheel commands of a recorded drive inside the simulator and compare
what the two sensor stacks reported tick by tick. With a real log the numbers
say how far the machine is from the model it was trained on; with two
simulator logs they say what a preset or a seed changes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from doomworm.hardware.calibration import Calibration
from doomworm.hardware.drive import DriveRow
from doomworm.hardware.link import SimLink
from doomworm.worlds import build_world


def replay_in_sim(meta: dict[str, Any], rows: Sequence[DriveRow]) -> list[DriveRow]:
    """Drive the recorded wheel sequence in a fresh simulator built from ``meta``.

    ``meta`` keys used: ``seed`` (world), ``maps``, ``task``, ``sensors``,
    ``sensor_seed``; missing ones take the benchmark defaults.
    """
    seed = int(meta.get("seed") or 3000)
    maps = str(meta.get("maps") or "apartment")
    task = str(meta.get("task") or "clean")
    sensors = str(meta.get("sensors") or "vacuum")
    sensor_seed = int(meta.get("sensor_seed") or 0)
    link = SimLink(build_world(seed, maps, task), sensors, sensor_seed, seed, maps, task)
    channels = link.reset()
    out: list[DriveRow] = []
    for row in rows:
        out.append(DriveRow(row.tick, row.wheels, dict(channels), None, link.truth()))
        channels = link.step(*row.wheels)
        if link.done:
            break
    return out


@dataclass
class ComparisonReport:
    """Per-channel error between two drives that sent the same wheels."""

    ticks: int
    channels: list[str]
    sum_sq: dict[str, float] = field(default_factory=dict)
    max_err: dict[str, float] = field(default_factory=dict)
    bumper_agreement: float = 1.0
    final_truth: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None

    def rmse(self, channel: str) -> float:
        """Root mean square error of one channel."""
        return math.sqrt(self.sum_sq[channel] / self.ticks) if self.ticks else math.nan

    def max_abs(self, channel: str) -> float:
        """Largest absolute error of one channel."""
        return self.max_err[channel]

    def pose_error_m(self, calibration: Calibration) -> float | None:
        """Distance between the final true poses, in metres (both logs need truth)."""
        if self.final_truth is None:
            return None
        a, b = self.final_truth
        return calibration.to_metres(math.dist(a[:2], b[:2]))

    def markdown(self, calibration: Calibration | None = None) -> str:
        """Report table."""
        cal = calibration or Calibration()
        lines = [
            f"ticks compared: {self.ticks}",
            f"bumper agreement: {self.bumper_agreement:.3f}",
        ]
        pose = self.pose_error_m(cal)
        if pose is not None:
            lines.append(f"final pose error: {pose:.3f} m ({cal.to_units(pose):.3f} u)")
        lines += ["", "| channel | rmse | max abs |", "|---|---|---|"]
        for name in self.channels:
            lines.append(f"| {name} | {self.rmse(name):.4f} | {self.max_abs(name):.4f} |")
        return "\n".join(lines) + "\n"


def compare_logs(a: Sequence[DriveRow], b: Sequence[DriveRow]) -> ComparisonReport:
    """Tick-by-tick channel error over the common prefix of two drives."""
    n = min(len(a), len(b))
    channels = [k for k in a[0].channels if k in b[0].channels] if n else []
    report = ComparisonReport(ticks=n, channels=channels)
    report.sum_sq = dict.fromkeys(channels, 0.0)
    report.max_err = dict.fromkeys(channels, 0.0)
    agree = 0
    bumper_keys = [k for k in ("bumper_left", "bumper_right") if k in channels]
    for ra, rb in zip(a[:n], b[:n], strict=True):
        for key in channels:
            err = ra.channels[key] - rb.channels[key]
            report.sum_sq[key] += err * err
            report.max_err[key] = max(report.max_err[key], abs(err))
        if all((ra.channels[k] > 0.5) == (rb.channels[k] > 0.5) for k in bumper_keys):
            agree += 1
    report.bumper_agreement = agree / n if n else 1.0
    if n and a[n - 1].truth is not None and b[n - 1].truth is not None:
        ta, tb = a[n - 1].truth, b[n - 1].truth
        assert ta is not None
        assert tb is not None
        report.final_truth = (ta, tb)
    return report
