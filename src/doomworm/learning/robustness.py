"""Robustness sweep (stage 22.1f): how a brain degrades when the assumed sensor numbers are wrong.

Every number of the ``car`` preset is an assumption until the machine is
measured. The sweep re-runs the benchmark with one parameter of the preset
changed at a time (worse than assumed, then worse again) and tabulates reward,
survival and collisions per cell, so the brain chosen for the car is the one
that degrades gracefully, not the one tuned to the guess.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace

from doomworm.environments.sensors import SensorConfig
from doomworm.episode import BrainLike
from doomworm.learning.benchmark import BenchmarkConfig, BenchmarkResult, run_benchmark

# base value first (the preset's), then worse and worse
CAR_GRID: dict[str, list[float | int]] = {
    "noise_sigma": [0.05, 0.15, 0.3],
    "dropout": [0.05, 0.15, 0.3],
    "odom_sigma": [0.1, 0.2, 0.4],
    "delay": [1, 2, 3],
    "beacon_fov": [math.radians(30.0), math.radians(20.0), math.radians(15.0)],
    "beacon_range": [5.0, 3.0, 2.0],
}

BrainFactory = Callable[[SensorConfig], BrainLike]


@dataclass
class SweepRow:
    """One cell of the sweep."""

    param: str
    value: float | int
    result: BenchmarkResult

    @property
    def label(self) -> str:
        """Human value (degrees for angles)."""
        if self.param == "beacon_fov":
            return f"{math.degrees(float(self.value)):.0f} deg"
        return f"{self.value:g}"


def sweep(
    factory: BrainFactory,
    name: str,
    base: SensorConfig,
    grid: Mapping[str, Sequence[float | int]],
    config: BenchmarkConfig,
) -> list[SweepRow]:
    """Baseline plus one row per (parameter, value) that differs from the base."""
    rows: list[SweepRow] = []
    cfg = replace(config, sensor_config=base)
    rows.append(SweepRow("base", 0, run_benchmark(factory(base), name, cfg)))
    for param, values in grid.items():
        for value in values:
            if getattr(base, param) == value:
                continue
            varied = SensorConfig(**(asdict(base) | {param: value}))
            cell = replace(config, sensor_config=varied)
            rows.append(SweepRow(param, value, run_benchmark(factory(varied), name, cell)))
    return rows


def sweep_table(rows: Sequence[SweepRow]) -> str:
    """Markdown: baseline first, then every varied cell with its drop against the baseline."""
    base = rows[0].result
    lines = [
        "| parameter | value | reward | vs base | survived | collisions | coverage |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        r = row.result
        delta = r.mean("reward") - base.mean("reward")
        label = "(preset)" if row.param == "base" else row.label
        lines.append(
            f"| {row.param} | {label} | {r.mean('reward'):.1f} ± {r.std('reward'):.1f} | "
            f"{delta:+.1f} | {r.mean('survived'):.2f} | {r.mean('collisions'):.1f} | "
            f"{r.mean('coverage'):.2f} |"
        )
    return "\n".join(lines) + "\n"


def worst_cells(rows: Sequence[SweepRow], n: int = 3) -> list[SweepRow]:
    """The cells with the largest reward drop against the baseline."""
    base = rows[0].result.mean("reward")
    varied = [r for r in rows[1:]]
    return sorted(varied, key=lambda r: r.result.mean("reward") - base)[:n]
