"""The benchmark every candidate brain goes through (Plan §3.3, stage 18)."""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

from broomworm.environments.maze import rooms_visited
from broomworm.environments.sensors import PRESETS, SensorConfig, SensorSuite
from broomworm.environments.worlds import build_world
from broomworm.episode import BrainLike, run_brain_episode
from broomworm.learning.reward import RewardConfig, RewardTracker

METRICS = (
    "reward", "ticks", "coverage", "food", "targets", "collisions", "damage",
    "dockings", "distance", "rooms", "survived",
)  # fmt: skip


@dataclass(frozen=True)
class BenchmarkConfig:
    """What every brain is measured on."""

    maps: str = "apartment"
    task: str = "clean"
    dangers: int = 0
    sensors: str = "vacuum"
    test_seeds: tuple[int, ...] = tuple(range(3000, 3006))
    steps: int = 800
    repeats: int = 3  # sensor-noise seeds per map
    sensor_config: SensorConfig | None = None  # overrides the ``sensors`` preset (sweeps)

    @property
    def episodes(self) -> int:
        """Total episodes per brain."""
        return len(self.test_seeds) * self.repeats


@dataclass
class EpisodeRow:
    """One episode's metrics."""

    seed: int
    repeat: int
    reward: float
    ticks: int
    coverage: float
    food: int
    targets: int
    collisions: int
    damage: int
    dockings: int
    distance: float
    rooms: int
    survived: int


@dataclass
class BenchmarkResult:
    """All episodes of one brain plus aggregates."""

    name: str
    config: BenchmarkConfig
    rows: list[EpisodeRow] = field(default_factory=list)

    def mean(self, metric: str) -> float:
        """Mean over episodes."""
        return statistics.fmean(getattr(r, metric) for r in self.rows) if self.rows else math.nan

    def std(self, metric: str) -> float:
        """Standard deviation over episodes (0 for a single episode)."""
        vals = [float(getattr(r, metric)) for r in self.rows]
        return statistics.pstdev(vals) if len(vals) > 1 else 0.0

    def summary(self) -> dict[str, float]:
        """Metric -> mean."""
        return {m: self.mean(m) for m in METRICS}

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready."""
        return {
            "name": self.name,
            "config": asdict(self.config),
            "summary": self.summary(),
            "std": {m: self.std(m) for m in METRICS},
            "rows": [asdict(r) for r in self.rows],
        }


def run_benchmark(
    brain: BrainLike,
    name: str,
    config: BenchmarkConfig | None = None,
    reward_config: RewardConfig | None = None,
    on_episode: Callable[[EpisodeRow], None] | None = None,
) -> BenchmarkResult:
    """Run every (map seed, repeat) episode for one brain."""
    cfg = config or BenchmarkConfig()
    if cfg.sensor_config is None and cfg.sensors not in PRESETS:
        raise ValueError(f"unknown sensor preset {cfg.sensors!r}")
    sensor_config = cfg.sensor_config or PRESETS[cfg.sensors]
    result = BenchmarkResult(name=name, config=cfg)
    for seed in cfg.test_seeds:
        for repeat in range(cfg.repeats):
            world = build_world(seed, cfg.maps, cfg.task, cfg.dangers, body=sensor_config.body)
            # Every preset goes through the suite: "ideal" is noiseless, not sensorless,
            # so planner-wrapped brains get odometry and bumper channels there too.
            suite = SensorSuite(sensor_config, seed=seed * 1000 + repeat)
            tracker = RewardTracker(reward_config or RewardConfig())
            trace = run_brain_episode(world, brain, cfg.steps, tracker, sensors=suite)
            distance = sum(math.dist((a.x, a.y), (b.x, b.y)) for a, b in pairwise(trace))
            row = EpisodeRow(
                seed=seed,
                repeat=repeat,
                reward=tracker.total,
                ticks=len(trace),
                coverage=world.coverage,
                food=world.food_eaten,
                targets=world.targets_reached,
                collisions=world.collisions,
                damage=world.damage_taken,
                dockings=world.dockings,
                distance=distance,
                rooms=rooms_visited(world, [(r.x, r.y) for r in trace]),
                survived=int(not world.dead),
            )
            result.rows.append(row)
            if on_episode is not None:
                on_episode(row)
    return result


LEADERBOARD_COLUMNS = (
    "reward",
    "coverage",
    "collisions",
    "damage",
    "dockings",
    "survived",
    "ticks",
)


def leaderboard(results: Sequence[BenchmarkResult]) -> str:
    """Markdown table sorted by mean reward, best first."""
    ranked = sorted(results, key=lambda r: -r.mean("reward"))
    columns = LEADERBOARD_COLUMNS
    head = "| # | brain | episodes | " + " | ".join(columns) + " |"
    sep = "|---|---|---|" + "---|" * len(columns)
    lines = [head, sep]
    for i, r in enumerate(ranked, 1):
        cells = []
        for m in columns:
            mean, std = r.mean(m), r.std(m)
            cells.append(
                f"{mean:.2f} ± {std:.2f}" if m in ("reward", "coverage") else f"{mean:.1f}"
            )
        lines.append(f"| {i} | {r.name} | {len(r.rows)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_result(result: BenchmarkResult, out_dir: Path) -> Path:
    """Write ``<name>.json`` and ``<name>.csv``; return the JSON path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result.name}.json"
    path.write_text(json.dumps(result.to_dict(), indent=2) + "\n")
    with (out_dir / f"{result.name}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(result.rows[0])) if result.rows else [])
        writer.writeheader()
        for r in result.rows:
            writer.writerow(asdict(r))
    return path


def load_results(out_dir: Path) -> list[BenchmarkResult]:
    """Read every ``*.json`` result in a directory (same config assumed)."""
    results = []
    for path in sorted(out_dir.glob("*.json")):
        data = json.loads(path.read_text())
        if "rows" not in data:
            continue
        cfg = BenchmarkConfig(
            **{**data["config"], "test_seeds": tuple(data["config"]["test_seeds"])}
        )
        res = BenchmarkResult(name=data["name"], config=cfg)
        fields = {f.name for f in dataclasses.fields(EpisodeRow)}
        res.rows = [
            EpisodeRow(**{k: v for k, v in row.items() if k in fields}) for row in data["rows"]
        ]
        results.append(res)
    return results


def write_leaderboard(out_dir: Path) -> str:
    """Rebuild ``leaderboard.md`` from every result in ``out_dir``."""
    table = leaderboard(load_results(out_dir))
    (out_dir / "leaderboard.md").write_text(table + "\n")
    return table
