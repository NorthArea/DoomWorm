#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Stage B2b: the Doom candidates at the training budget axis C6 paid for.

B2 trained on three maps for 25 generations, and the result is visible in the
counters: the evolved brains fire 50 rounds for 0.0-0.2 kills while the
hand-written floor takes 0.5 kills on 5.7 shots. Axis C6 already showed the
worm needs far more experience than the A2 protocol grants it, so this lane
raises the two knobs that cost nothing but time -- 12 training maps instead of
3 (seeds 100-111) and 60 generations instead of 25 -- and leaves everything
else at the B2 protocol: level doom4 (the first level with the gun), ideal
sensors, no planner, 600 steps.

One repeat per map: the ideal preset is noiseless, so a second repeat would be
the same episode.

    uv run scripts/doom_b2plus_lane.py --pilot     # seed 0, worm + curriculum
    uv run scripts/doom_b2plus_lane.py             # three seeds, four rows

The pilot exists so that four hours are spent only if one run shows the
counters moving: the question it answers is "does more budget buy aiming?".
Every row is benchmarked on the 12 unseen maps of its level right after
training, and a run already holding a `.meta.json` is skipped.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs/doom_b2plus"
LANE_LOG = OUT / "lane.log"
CURRICULUM = "docs/brains/a1/worm_evolved_random.json"

LEVEL = "doom4"
EVAL_MAPS = ("doom4", "doom6", "vizdoom4")  # same level, a harder one, and the engine
GENERATIONS = "60"
TRAIN_MAPS = "12"
WORKERS = "6"

ROWS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("worm", "worm", ()),
    ("worm_from_worm_evolved_random", "worm", ("--init-brain", CURRICULUM)),
    ("rnn", "rnn", ()),
)
PILOT_ROWS = ROWS[:2]


def stamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(line: str) -> None:
    LANE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with LANE_LOG.open("a") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def run(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"=== START {stamp()} === {' '.join(cmd)}")
    with log_path.open("a") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
    log(f"=== END {stamp()} rc={rc} ===")
    tail = [ln for ln in log_path.read_text().splitlines() if "best train fitness" in ln]
    if tail:
        log(f"    {tail[-1].strip()}")
    return rc


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for(pid: int) -> None:
    """Hold the lane until another lane's process is gone (one heavy run at a time)."""
    log(f"=== WAITING for pid {pid} === {stamp()}")
    while alive(pid):
        time.sleep(60)
    log(f"=== pid {pid} finished, starting === {stamp()}")


def train_and_benchmark(seed: int, name: str, candidate: str, extra: tuple[str, ...]) -> None:
    out = OUT / f"seed{seed}" / f"{name}.json"
    if out.with_suffix(".meta.json").exists():
        log(f"    have {out.relative_to(ROOT)}, skipping training")
    else:
        rc = run(
            ["uv", "run", "doomworm", "evolve", "--candidate", candidate, *extra,
             "--layer", "none", "--maps", LEVEL, "--task", "doom", "--sensors", "ideal",
             "--steps", "600", "--train-seeds", TRAIN_MAPS, "--generations", GENERATIONS,
             "--workers", WORKERS, "--seed", str(seed), "--out", str(out.relative_to(ROOT))],
            out.with_suffix(".log"),
        )  # fmt: skip
        if rc != 0:
            log(f"    CRASH: see {out.with_suffix('.log').relative_to(ROOT)}")
            return
    for eval_map in EVAL_MAPS:
        bench_dir = OUT / "benchmark" / f"seed{seed}" / f"eval_{eval_map}"
        if (bench_dir / f"{name}.csv").exists():
            continue
        run(
            ["uv", "run", "doomworm", "benchmark", "--brain", str(out.relative_to(ROOT)),
             "--maps", eval_map, "--task", "doom", "--sensors", "ideal", "--steps", "600",
             "--test-seeds", "12", "--repeats", "1", "--out-dir", str(bench_dir.relative_to(ROOT))],
            out.with_suffix(".bench.log"),
        )  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot", action="store_true", help="seed 0, worm and curriculum worm only")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--after-pid", type=int, default=None, help="wait for this process first")
    ap.add_argument("--no-detach", action="store_true")
    args = ap.parse_args()

    if not args.no_detach:
        os.setsid()
    if args.after_pid:
        wait_for(args.after_pid)

    rows = PILOT_ROWS if args.pilot else ROWS
    seeds = (0,) if args.pilot else tuple(range(args.seeds))
    started = time.time()
    log(f"=== B2+ LANE START {stamp()} === {len(rows) * len(seeds)} runs, {GENERATIONS} gen "
        f"x {TRAIN_MAPS} maps on {LEVEL}")  # fmt: skip
    for seed in seeds:
        for name, candidate, extra in rows:
            train_and_benchmark(seed, name, candidate, extra)
    log(f"=== B2+ LANE DONE {stamp()} === {(time.time() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
