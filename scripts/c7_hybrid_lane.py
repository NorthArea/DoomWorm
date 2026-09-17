#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Axis C7 (stage 23.3): the hybrid against the parts it is made of.

Protocol of axis C6 so the rows compare directly (`docs/findings.md`): car
preset, layer v2 (`needs`), 12 training maps x 2 noise repeats, evolution
40 x 25, three seeds, then the standard car benchmark (18 unseen episodes).

Rows trained here:

  * ``hybrid``  -- the seed's C6 curriculum worm, frozen, under a fresh small
    net that sees its wheels as three extra channels;
  * ``rnn``     -- the same net without the worm, the control that says whether
    the connectome contributes anything at all.

The worm (44.4 +- 5.6) and PPO (37.5 +- 8.4) rows of C6 are the other two
comparisons and are not retrained.

Skips a run whose ``.meta.json`` already exists, so it can be re-run to fill
gaps. Detaches into its own session; one line per command in the lane log.
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
OUT = ROOT / "runs/c7"
LANE_LOG = OUT / "lane.log"
REFLEX = "docs/brains/car_12maps/seed{seed}/worm_from_worm_evolved_random.json"

SEEDS = (0, 1, 2)
WORKERS = "4"  # the track-B lane holds six of the machine's cores
TRAIN = (
    "--sensors", "car",
    "--layer", "needs",
    "--train-seeds", "12",
    "--train-repeats", "2",
)  # fmt: skip
BENCH = ("--sensors", "car", "--planner", "needs")


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
    for marker in ("best train fitness", "reward"):
        hits = [ln for ln in log_path.read_text().splitlines() if marker in ln]
        if hits:
            log(f"    {hits[-1].strip()}")
    return rc


def jobs(seed: int) -> tuple[tuple[str, list[str]], ...]:
    """The two rows of the axis for one seed."""
    return (
        ("hybrid", ["--candidate", "hybrid", "--init-brain", REFLEX.format(seed=seed)]),
        ("rnn", ["--candidate", "rnn"]),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-detach", action="store_true")
    args = ap.parse_args()

    todo = [
        f"{name} seed{seed}"
        for seed in SEEDS
        for name, _ in jobs(seed)
        if not (OUT / f"seed{seed}" / f"{name}.meta.json").exists()
    ]
    if args.dry_run:
        print(f"{len(todo)} runs left: " + ", ".join(todo))
        return 0
    if not args.no_detach:
        os.setsid()

    started = time.time()
    log(f"=== C7 LANE START {stamp()} === {len(todo)} runs")
    for seed in SEEDS:
        for name, flags in jobs(seed):
            out = OUT / f"seed{seed}" / f"{name}.json"
            if out.with_suffix(".meta.json").exists():
                log(f"    have {out.relative_to(ROOT)}, skipping")
            else:
                rc = run(
                    ["uv", "run", "doomworm", "evolve", *flags, *TRAIN,
                     "--workers", WORKERS, "--seed", str(seed),
                     "--out", str(out.relative_to(ROOT))],
                    out.with_suffix(".log"),
                )  # fmt: skip
                if rc != 0:
                    log(f"    CRASH: see {out.with_suffix('.log').relative_to(ROOT)}")
                    continue
            run(
                ["uv", "run", "doomworm", "benchmark", "--brain", str(out.relative_to(ROOT)),
                 "--name", f"{name}_c7_seed{seed}", *BENCH,
                 "--out-dir", str((OUT / "benchmark").relative_to(ROOT))],
                out.with_suffix(".bench.log"),
            )  # fmt: skip
    log(f"=== C7 LANE DONE {stamp()} === {(time.time() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
