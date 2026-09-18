"""Turn a lane's raw benchmark rows into a table that outlives `runs/`.

    uv run python scripts/archive_runs.py

`runs/` is the scratch desk: brains, csv rows, lane logs. Anything that backs a
published number has to survive it. This walks the benchmark directories a lane
wrote, aggregates every row over seeds, and writes one markdown table per lane
into `docs/doom/results/`, next to a copy of the lane log. What it cannot do is
invent context, so each table says which lane, which levels and how many seeds
it came from and nothing more.
"""

from __future__ import annotations

import csv
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = ROOT / "docs/doom/results"
COLUMNS = (
    "reward", "exited", "kills", "hits", "shots", "damage", "survived", "ticks", "collisions",
)  # fmt: skip

# lane -> (benchmark directory, the lane log to keep beside the table)
LANES: dict[str, tuple[str, str]] = {
    "b2_bakeoff": ("benchmark_doom", "doom/train.log"),
    "b2d_shooting": ("benchmark_doom_b2d", "doom_b2d/train.log"),
    "b2d_asg_superseded": ("benchmark_doom_b2d_asg", "doom_b2d_asg/train.log"),
    "memory": ("memory/benchmark", "memory/lane.log"),
    "squeeze_genome": ("squeeze/benchmark", "squeeze/lane.log"),
    "squeeze_search": ("squeeze2/benchmark", "squeeze2/lane.log"),
    "hybrid_c7": ("c7/benchmark", "c7/lane.log"),
    "budget_pilot": ("doom_b2plus/benchmark", "doom_b2plus/lane.log"),
}


def rows_of(csv_path: Path) -> dict[str, float]:
    """Mean of every column over the episodes in one csv."""
    values = list(csv.DictReader(csv_path.open()))
    out: dict[str, float] = {}
    for column in COLUMNS:
        numbers = [float(v[column]) for v in values if column in v and v[column] != ""]
        if numbers:
            out[column] = st.fmean(numbers)
    return out


def collect(base: Path) -> dict[tuple[str, str], list[dict[str, float]]]:
    """(training group, evaluation level) -> one dict of means per seed."""
    found: dict[tuple[str, str], list[dict[str, float]]] = {}
    for csv_path in sorted(base.rglob("*.csv")):
        parts = csv_path.relative_to(base).parts
        level = next((p[len("eval_") :] for p in parts if p.startswith("eval_")), "?")
        group = csv_path.stem
        found.setdefault((group, level), []).append(rows_of(csv_path))
    return found


def table(found: dict[tuple[str, str], list[dict[str, float]]]) -> str:
    levels = sorted({level for _, level in found})
    lines = []
    for level in levels:
        lines.append(f"\n### Evaluated on `{level}`\n")
        lines.append("| brain | seeds | reward | exits | kills | hits | shots | survived |")
        lines.append("|---|---|---|---|---|---|---|---|")
        entries = [(g, seeds) for (g, lv), seeds in found.items() if lv == level]
        entries.sort(key=lambda kv: -st.fmean(s.get("reward", 0.0) for s in kv[1]))
        for group, seeds in entries:
            rewards = [s.get("reward", 0.0) for s in seeds]
            spread = f" ± {st.stdev(rewards):.1f}" if len(rewards) > 1 else ""

            def mean(column: str, seeds: list[dict[str, float]] = seeds) -> float:
                return st.fmean(s.get(column, 0.0) for s in seeds)

            lines.append(
                f"| {group} | {len(seeds)} | {st.fmean(rewards):.2f}{spread} | "
                f"{mean('exited'):.2f} | {mean('kills'):.2f} | {mean('hits'):.2f} | "
                f"{mean('shots'):.1f} | {mean('survived'):.2f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    if not RUNS.exists():
        print("no runs/ to archive")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    written = 0
    for lane, (bench, log) in LANES.items():
        base = RUNS / bench
        if not base.exists():
            continue
        found = collect(base)
        if not found:
            continue
        target = OUT / lane
        target.mkdir(parents=True, exist_ok=True)
        header = (
            f"# {lane}\n\n"
            f"Aggregated from `runs/{bench}` on the day `runs/` was cleared. Every number "
            f"is the mean over the episodes of a row, then over seeds; the spread is across "
            f"seeds. The brains that produced these rows are in `docs/doom/brains/`.\n"
        )
        (target / "summary.md").write_text(header + table(found))
        source_log = RUNS / log
        if source_log.exists():
            (target / "lane.log").write_text(source_log.read_text())
        written += 1
        print(f"  {lane}: {len(found)} rows -> {target.relative_to(ROOT)}")
    print(f"archived {written} lanes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
