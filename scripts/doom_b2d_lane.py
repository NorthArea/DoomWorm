#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Stage B2d: the worm rows retrained with a reachable trigger and a target it can turn to.

The B2 rows were trained while `aim` reached only the amphid pair, so the
trigger group -- which lives in the pharyngeal island -- never saw the target
and the real connectome could not learn to shoot at all (its shuffled control
could: shuffling gives the island 399 crossing connections).

Two things were wrong, both ours, both measured on untrained brains first:

  * the trigger group lives in the pharyngeal island, whose only door is RIP,
    so `aim` never reached it (fire 0.134 with an enemy on the gun line against
    0.157 with none -- pure noise). `aim` is now delivered to RIP as well: the
    same untrained worm answers 0.906 against 0.143.
  * the gun is bolted to the chassis, so aiming is turning, and the enemy only
    existed on `danger_*` -> ASH, the escape pathway. The new `prey_*` channels
    read the same monster as a lateralised attractant on ASG, chosen by
    measuring which of the 31 free sensory pairs actually turns the untrained
    body the right way. Now an enemy on the left turns the body left.

The worm family is retrained under the unchanged B2 protocol: doom4 and doom6, three seeds,
evolution 40 x 25 on maps 100-102, ideal sensors, no planner, 600 steps.

Only the worm family is retrained: rnn, ncp and PPO never touch the connectome,
so their published rows stand. Same bookkeeping as the B2 lane:

  * evolve  -- a brain counts as trained when `<name>.meta.json` sits next to
    `<name>.json`; the bare `.json` is a mid-run checkpoint and is retrained.
  * benchmark on the simulator levels doom1..doom6 and, for stage B4, on the
    engine levels vizdoom1..vizdoom6 -- one eval counts as done when its csv
    exists under `runs/benchmark_doom/train_<level>/seed<seed>/eval_<map>/`.

It detaches into its own session so a closing terminal cannot take it down,
appends a START/END line per command to `runs/doom/train.log`, and never
touches a finished artefact.
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
TRAIN_LOG = ROOT / "runs/doom_b2d/train.log"

LEVELS = ("doom4", "doom6")
SEEDS = (0, 1, 2)
# name -> (candidate, extra args); order is the one the original lane used
EVOLVE_JOBS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("worm", "worm", ()),
    (
        "worm_from_worm_evolved_random",
        "worm",
        ("--init-brain", "docs/brains/a1/worm_evolved_random.json"),
    ),
    ("worm_random", "worm_random", ()),
    ("worm_shuffled", "worm_shuffled", ()),
)
# ppo is trained by its own lane; it still takes part in the benchmarks
BENCH_NAMES = tuple(name for name, _, _ in EVOLVE_JOBS)  # the nets are untouched by B2c
SIM_MAPS = tuple(f"doom{n}" for n in range(1, 7))
ENGINE_MAPS = tuple(f"vizdoom{n}" for n in range(1, 7))

STEPS = "600"
TEST_SEEDS = "12"
WORKERS = "10"


def stamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(line: str) -> None:
    with TRAIN_LOG.open("a") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def run(cmd: list[str], log_path: Path, *, announce: bool = True) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pretty = " ".join(cmd)
    if announce:
        log(f"=== START {stamp()} === {pretty}")
    with log_path.open("a") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
    if announce:
        log(f"=== END {stamp()} rc={rc} === {pretty}")
        best = [ln for ln in log_path.read_text().splitlines() if "best train fitness" in ln]
        if best:
            log(f"    {best[-1].strip()}")
        if rc != 0:
            log(f"    CRASH: see {log_path.relative_to(ROOT)}")
    return rc


def brain_path(level: str, seed: int, name: str) -> Path:
    return ROOT / f"runs/doom_b2d/{level}/seed{seed}/{name}.json"


def trained(level: str, seed: int, name: str) -> bool:
    brain = brain_path(level, seed, name)
    return brain.exists() and brain.with_suffix(".meta.json").exists()


def eval_csv(level: str, seed: int, name: str, eval_map: str) -> Path:
    return ROOT / f"runs/benchmark_doom_b2d/train_{level}/seed{seed}/eval_{eval_map}/{name}.csv"


def evolve(level: str, seed: int, name: str, candidate: str, extra: tuple[str, ...]) -> None:
    out = brain_path(level, seed, name)
    cmd = [
        "uv",
        "run",
        "doomworm",
        "evolve",
        "--candidate",
        candidate,
        "--layer",
        "none",
        "--maps",
        level,
        "--task",
        "doom",
        "--sensors",
        "ideal",
        "--steps",
        STEPS,
        "--workers",
        WORKERS,
        "--seed",
        str(seed),
        *extra,
        "--out",
        str(out.relative_to(ROOT)),
    ]
    run(cmd, out.with_suffix(".log"))


def benchmark(level: str, seed: int, name: str, maps: tuple[str, ...]) -> None:
    brain = brain_path(level, seed, name)
    if not brain.exists():
        log(f"SKIP benchmark: {brain.relative_to(ROOT)} missing")
        return
    for eval_map in maps:
        csv = eval_csv(level, seed, name, eval_map)
        if csv.exists():
            continue
        cmd = [
            "uv",
            "run",
            "doomworm",
            "benchmark",
            "--brain",
            str(brain.relative_to(ROOT)),
            "--maps",
            eval_map,
            "--task",
            "doom",
            "--sensors",
            "ideal",
            "--steps",
            STEPS,
            "--test-seeds",
            TEST_SEEDS,
            "--repeats",
            "1",
            "--out-dir",
            str(csv.parent.relative_to(ROOT)),
        ]
        rc = run(cmd, brain.with_suffix(".bench.log"), announce=False)
        if rc != 0 or not csv.exists():
            log(f"FAIL benchmark rc={rc}: {name} {level}/seed{seed} -> {eval_map}")


def plan() -> list[str]:
    todo: list[str] = []
    for level in LEVELS:
        for seed in SEEDS:
            for name, _, _ in EVOLVE_JOBS:
                if not trained(level, seed, name):
                    todo.append(f"evolve {level}/seed{seed}/{name}")
            for name in BENCH_NAMES:
                missing = [
                    m for m in SIM_MAPS + ENGINE_MAPS if not eval_csv(level, seed, name, m).exists()
                ]
                if missing:
                    todo.append(f"bench  {level}/seed{seed}/{name}: {' '.join(missing)}")
    return todo


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="print what is missing and exit")
    ap.add_argument("--no-detach", action="store_true", help="stay in the caller's session")
    args = ap.parse_args()

    if args.dry_run:
        todo = plan()
        print(f"{len(todo)} jobs left")
        for line in todo:
            print("  " + line)
        return 0

    if not args.no_detach:
        os.setsid()

    started = time.time()
    log(f"=== RESUME START {stamp()} === {len(plan())} jobs")

    for level in LEVELS:
        for seed in SEEDS:
            for name, candidate, extra in EVOLVE_JOBS:
                if trained(level, seed, name):
                    log(f"    have {level}/seed{seed}/{name}, skipping training")
                else:
                    evolve(level, seed, name, candidate, extra)
                benchmark(level, seed, name, SIM_MAPS)

    # Stage B4: every trained brain, unchanged, in the Doom engine.
    log(f"=== ENGINE TRANSFER START {stamp()} ===")
    for level in LEVELS:
        for seed in SEEDS:
            for name in BENCH_NAMES:
                benchmark(level, seed, name, ENGINE_MAPS)

    log(f"=== RESUME DONE {stamp()} === {(time.time() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
