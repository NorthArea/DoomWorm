#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Stage 18: the last squeeze on the worm — the search, not the brain.

Everything here changes how the weights are found, never the network. The
connectome is too well connected for path-reachability to narrow the search
(within two hops of the sensors, 99.3 % of synapses can already reach a motor
group), so the useful cut is the interface:

    all         5905 synapses, as before
    interface   1819: those leaving a sensory neuron or entering a motor group
    sensory      506: only those leaving a sensory neuron

Evolution gets 1000 evaluations either way, so the smaller the search space the
more each one is worth. The fourth row raises the motor turn gain, which the
untrained probe showed is what brings the body round at all (best bearing 65
degrees at gain 6, 26 at gain 24).

Trained and benchmarked on doom4, three seeds, twelve unseen maps.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs/squeeze"
LOG = OUT / "lane.log"
SEEDS = (0, 1, 2)
ROWS = (
    ("all", []),
    ("interface", ["--trainable", "interface"]),
    ("sensory", ["--trainable", "sensory"]),
)
EVAL = ("doom4", "doom6", "e1m1")


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
    log(f"=== SQUEEZE LANE START {stamp()} === {len(ROWS) * len(SEEDS)} runs")
    for seed in SEEDS:
        for name, flags in ROWS:
            out = OUT / f"seed{seed}" / f"{name}.json"
            if not out.with_suffix(".meta.json").exists():
                rc = run(
                    ["uv", "run", "doomworm", "evolve", "--candidate", "worm", *flags,
                     "--maps", "doom4", "--task", "doom", "--sensors", "ideal", "--steps", "600",
                     "--workers", "10", "--seed", str(seed), "--out", str(out.relative_to(ROOT))],
                    out.with_suffix(".log"),
                )  # fmt: skip
                if rc != 0:
                    log(f"    CRASH: {out.name}")
                    continue
            for eval_map in EVAL:
                bench = OUT / "benchmark" / f"seed{seed}" / f"eval_{eval_map}"
                if (bench / f"{name}.csv").exists():
                    continue
                seeds = "3" if eval_map.startswith("e") else "12"
                run(
                    ["uv", "run", "doomworm", "benchmark", "--brain", str(out.relative_to(ROOT)),
                     "--name", name, "--maps", eval_map, "--task", "doom", "--sensors", "ideal",
                     "--steps", "600", "--test-seeds", seeds, "--repeats", "1",
                     "--out-dir", str(bench.relative_to(ROOT))],
                    out.with_suffix(".bench.log"),
                )  # fmt: skip
    log(f"=== SQUEEZE LANE DONE {stamp()} === {(time.time() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
