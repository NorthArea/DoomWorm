#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Stage 16: does a place memory outside the brain buy the exit?

No trained brain has ever reached one. The memory layer gives the worm the
sense it lacks -- where it has *not* been, as a gradient in the same form as
the food smell -- and the brain still decides what to do with it.

Four rows, three seeds, trained on doom4 (a partition with one doorway, round
obstacles, a monster and the gun) and benchmarked on doom1..doom6 under the
condition they were trained in:

    worm             worm+memory
    worm_shuffled    worm_shuffled+memory

The shuffled control answers the obvious objection: if the memory helps, does
it help *this* wiring or any wiring of the same shape?
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs/memory"
LOG = OUT / "lane.log"
SEEDS = (0, 1, 2)
ROWS = (("worm", False), ("worm", True), ("worm_shuffled", False), ("worm_shuffled", True))
EVAL_MAPS = ("doom1", "doom2", "doom3", "doom4", "doom5", "doom6")


def stamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(line: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def run(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"=== START {stamp()} === {' '.join(cmd)}")
    with log_path.open("a") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
    log(f"=== END {stamp()} rc={rc} ===")
    best = [ln for ln in log_path.read_text().splitlines() if "best train fitness" in ln]
    if best:
        log(f"    {best[-1].strip()}")
    return rc


def main() -> int:
    if "--no-detach" not in sys.argv:
        os.setsid()
    started = time.time()
    log(f"=== MEMORY LANE START {stamp()} === {len(ROWS) * len(SEEDS)} runs")
    for seed in SEEDS:
        for candidate, memory in ROWS:
            name = candidate + ("_memory" if memory else "")
            out = OUT / f"seed{seed}" / f"{name}.json"
            flags = ["--memory"] if memory else []
            if not out.with_suffix(".meta.json").exists():
                rc = run(
                    ["uv", "run", "doomworm", "evolve", "--candidate", candidate, *flags,
                     "--maps", "doom4", "--task", "doom", "--sensors", "ideal", "--steps", "600",
                     "--workers", "10", "--seed", str(seed), "--out", str(out.relative_to(ROOT))],
                    out.with_suffix(".log"),
                )  # fmt: skip
                if rc != 0:
                    log(f"    CRASH: {out.name}")
                    continue
            for eval_map in EVAL_MAPS:
                bench = OUT / "benchmark" / f"seed{seed}" / f"eval_{eval_map}"
                if (bench / f"{name}.csv").exists():
                    continue
                run(
                    ["uv", "run", "doomworm", "benchmark", "--brain", str(out.relative_to(ROOT)),
                     "--name", name, *flags, "--maps", eval_map, "--task", "doom",
                     "--sensors", "ideal", "--steps", "600", "--test-seeds", "12",
                     "--repeats", "1", "--out-dir", str(bench.relative_to(ROOT))],
                    out.with_suffix(".bench.log"),
                )  # fmt: skip
    log(f"=== MEMORY LANE DONE {stamp()} === {(time.time() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
