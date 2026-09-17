"""Track B report: tables from the Doom benchmark rows (simulator and engine).

    uv run python scripts/doom_report.py runs/benchmark_doom --out docs/results/b/b2_doom_<date>.md

Reads ``train_<level>/seed<S>/eval_<level>/*.json`` (three seeds per candidate),
``floors/eval_<level>/`` and ``a2_transfer/eval_<level>/`` and writes: the
same-level leaderboards with the floors and the A2 brains appended, a transfer
matrix per training level over every evaluation level (simulator ``doom1..6``
and engine ``vizdoom1..6``), and the simulator-vs-engine gap per candidate.
Every number is a mean over the 12 unseen maps of a row, then over seeds.
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

from doomworm.learning.benchmark import BenchmarkResult, load_results

COLUMNS = (
    "reward",
    "exited",
    "kills",
    "hits",
    "shots",
    "damage",
    "survived",
    "ticks",
    "collisions",
)
SIM = tuple(f"doom{n}" for n in range(1, 7))
ENGINE = tuple(f"vizdoom{n}" for n in range(1, 7))
ORDER = (
    "worm",
    "worm_from_worm_evolved_random",
    "worm_random",
    "worm_shuffled",
    "rnn",
    "ncp",
    "ppo",
)


def by_seed(root: Path, train: str, eval_level: str) -> dict[str, list[BenchmarkResult]]:
    """Candidate -> one result per seed directory that has the row."""
    groups: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for seed_dir in sorted(root.glob(f"train_{train}/seed*")):
        d = seed_dir / f"eval_{eval_level}"
        if d.is_dir():
            for r in load_results(d):
                groups[r.name].append(r)
    return dict(groups)


def singles(root: Path, folder: str, eval_level: str) -> list[BenchmarkResult]:
    d = root / folder / f"eval_{eval_level}"
    return load_results(d) if d.is_dir() else []


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), (statistics.pstdev(values) if len(values) > 1 else 0.0)


def cell(values: list[float], digits: int = 1) -> str:
    if not values:
        return "n/a"
    m, s = mean_std(values)
    return f"{m:.{digits}f} ± {s:.{digits}f}" if len(values) > 1 else f"{m:.{digits}f}"


def leaderboard(root: Path, train: str, eval_level: str) -> str:
    rows: list[tuple[float, str]] = []
    groups = by_seed(root, train, eval_level)
    for name in ORDER:
        results = groups.get(name, [])
        if not results:
            continue
        rewards = [r.mean("reward") for r in results]
        cells = [cell(rewards)] + [
            cell([r.mean(c) for r in results], 2 if c in ("exited", "survived", "kills") else 1)
            for c in COLUMNS[1:]
        ]
        rows.append(
            (mean_std(rewards)[0], f"| {name} | {len(results)} | " + " | ".join(cells) + " |")
        )
    for folder, tag in (("floors", "floor"), ("a2_transfer", "A2 brain, no retraining")):
        for r in singles(root, folder, eval_level):
            cells = [f"{r.mean('reward'):.1f}"] + [
                f"{r.mean(c):.2f}" if c in ("exited", "survived", "kills") else f"{r.mean(c):.1f}"
                for c in COLUMNS[1:]
            ]
            rows.append((r.mean("reward"), f"| {r.name} ({tag}) | 1 | " + " | ".join(cells) + " |"))
    head = "| brain | seeds | " + " | ".join(COLUMNS) + " |\n|---|---|" + "---|" * len(COLUMNS)
    body = "\n".join(line for _, line in sorted(rows, key=lambda t: -t[0]))
    return head + "\n" + body


def matrix(root: Path, train: str, levels: tuple[str, ...]) -> str:
    head = "| brain | " + " | ".join(levels) + " |\n|---|" + "---|" * len(levels)
    lines = [head]
    for name in ORDER:
        cells = []
        for lvl in levels:
            results = by_seed(root, train, lvl).get(name, [])
            cells.append(cell([r.mean("reward") for r in results]))
        if any(c != "n/a" for c in cells):
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    for folder, tag in (("floors", "floor"), ("a2_transfer", "A2")):
        names = sorted({r.name for lvl in levels for r in singles(root, folder, lvl)})
        for nm in names:
            cells = []
            for lvl in levels:
                hit = [r for r in singles(root, folder, lvl) if r.name == nm]
                cells.append(f"{hit[0].mean('reward'):.1f}" if hit else "n/a")
            lines.append(f"| {nm} ({tag}) | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def gap(root: Path, train: str) -> str:
    head = "| brain | " + " | ".join(f"{s} -> {e}" for s, e in zip(SIM, ENGINE, strict=True)) + " |"
    lines = [head, "|---|" + "---|" * len(SIM)]
    for name in (*ORDER, "doomguy", "follower", "a2_worm_curriculum", "a2_ppo", "a2_rnn"):
        cells = []
        for s, e in zip(SIM, ENGINE, strict=True):
            if name in ORDER:
                a = [r.mean("reward") for r in by_seed(root, train, s).get(name, [])]
                b = [r.mean("reward") for r in by_seed(root, train, e).get(name, [])]
            else:
                folder = "floors" if name in ("doomguy", "follower") else "a2_transfer"
                a = [r.mean("reward") for r in singles(root, folder, s) if r.name == name]
                b = [r.mean("reward") for r in singles(root, folder, e) if r.name == name]
            if a and b:
                cells.append(f"{statistics.fmean(a):.1f} -> {statistics.fmean(b):.1f}")
            else:
                cells.append("n/a")
        if any(c != "n/a" for c in cells):
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    root = args.root
    parts = ["# Track B: Doom benchmark rows (generated by scripts/doom_report.py)\n"]
    for train in ("doom4", "doom6"):
        sections = (
            (
                f"Trained on {train}, evaluated on {train} (simulator)",
                leaderboard(root, train, train),
            ),
            (
                f"Trained on {train}: transfer over the simulator levels (reward)",
                matrix(root, train, SIM),
            ),
            (
                f"Trained on {train}: the same brains in the Doom engine (reward)",
                matrix(root, train, ENGINE),
            ),
            (f"Trained on {train}: simulator -> engine gap (reward)", gap(root, train)),
            (
                f"Trained on {train}, evaluated on viz{train} (engine)",
                leaderboard(root, train, "viz" + train),
            ),
        )
        parts.extend(f"## {title}\n\n{table}\n" for title, table in sections)
    text = "\n".join(parts)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
