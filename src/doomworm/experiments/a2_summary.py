"""Stage 21.7: summarise the bake-off over evolution / PPO seeds.

Reads benchmark result JSONs from several directories (one per seed) and
prints, per candidate, the mean and spread over seeds of every leaderboard
metric, plus the per-seed rewards, so the decision rests on the spread
between seeds and not on one run (Plan §20.4: at least 3 seeds).

    python -m doomworm.experiments.a2_summary runs/benchmark_a2 runs/benchmark_a2/seed1 ...
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from doomworm.learning.benchmark import LEADERBOARD_COLUMNS, BenchmarkResult, load_results


def collect(dirs: Sequence[Path]) -> dict[str, list[BenchmarkResult]]:
    """Group results by row name across directories (one result per seed)."""
    by_name: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for d in dirs:
        for r in load_results(d):
            by_name[r.name].append(r)
    return dict(by_name)


def summary_table(groups: dict[str, list[BenchmarkResult]]) -> str:
    """Markdown: one row per candidate, mean ± std over seeds, sorted by mean reward."""
    lines = [
        "| # | brain | seeds | reward (mean ± std over seeds) | per-seed reward | "
        + " | ".join(c for c in LEADERBOARD_COLUMNS if c != "reward")
        + " |",
        "|---|---|---|---|---|" + "---|" * (len(LEADERBOARD_COLUMNS) - 1),
    ]
    rows = []
    for name, results in groups.items():
        rewards = [r.mean("reward") for r in results]
        mean = statistics.fmean(rewards)
        std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
        others = [
            statistics.fmean(r.mean(c) for r in results)
            for c in LEADERBOARD_COLUMNS
            if c != "reward"
        ]
        rows.append((mean, std, name, len(results), rewards, others))
    for i, (mean, std, name, n, rewards, others) in enumerate(sorted(rows, reverse=True), 1):
        per_seed = ", ".join(f"{v:.1f}" for v in rewards)
        cells = " | ".join(f"{v:.2f}" for v in others)
        lines.append(f"| {i} | {name} | {n} | {mean:.2f} ± {std:.2f} | {per_seed} | {cells} |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dirs", nargs="+", type=Path, help="benchmark result directories")
    parser.add_argument("--out", type=Path, default=None, help="write the markdown here too")
    args = parser.parse_args(argv)
    table = summary_table(collect(args.dirs))
    print(table)
    if args.out is not None:
        args.out.write_text(table + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
