"""Stage 26: what the hand-written floor knows, one reflex at a time.

    uv run python scripts/ablation_lane.py --leave-one-out
    uv run python scripts/ablation_lane.py --only-one

Stage 25 left the project with one fact worth chasing: the hand-written floor is
the only thing here that ever leaves the band of flailing at random -- 59 % of
what `doom6` allows, against every trained brain's zero-or-less. So the question
is not another optimiser. It is *what that floor knows*, stated as numbers.

Seven reflexes, and two ways to price each one.

    leave one out   remove a single reflex and let the tick fall through to the
                    next. What it costs is what that piece of knowledge is worth
                    on top of everything else.

    only one        keep a single reflex and remove the other six. What it scores
                    is how far that piece gets you alone -- which is the form the
                    question takes for a brain that has to grow one behaviour at
                    a time.

Both run down the same path every published row came from, at the same 12 seeds,
2 repeats and 600 ticks, so the numbers sit beside the rest of findings.md
without an asterisk.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from doomworm.doomguy import REFLEXES

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs/ablation"
LEVELS = ("doom1", "doom2", "doom4", "doom6")
STEPS, SEEDS, REPEATS = "600", "12", "2"
CEILING = {"doom1": 22.0, "doom2": 22.0, "doom4": 34.0, "doom6": 46.0}  # stage 25, reachable


def bench(level: str, ablate: tuple[str, ...], tag: str) -> float:
    """One arm on one level; returns its mean reward."""
    cmd = [
        "wormlab", "benchmark",
        "--maps", level, "--task", "doom", "--sensors", "ideal",
        "--scripted", "doomguy", "--name", tag,
        "--steps", STEPS, "--test-seeds", SEEDS, "--repeats", REPEATS,
        "--out-dir", str(OUT / level),
    ]  # fmt: skip
    if ablate:
        cmd += ["--ablate", *ablate]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout
    # by name, never by rank: the summary lists every arm already in the directory
    row = next((ln for ln in out.splitlines() if f"| {tag} |" in ln), "")
    found = re.search(r"\| (-?\d+\.\d+) ±", row)
    return float(found.group(1)) if found else float("nan")


def table(arms: list[tuple[str, tuple[str, ...]]], title: str) -> None:
    """One row per arm, one column per level, against the full floor."""
    print(f"\n{title}   ({SEEDS} seeds x {REPEATS} repeats x {STEPS} ticks)\n")
    print("  arm            " + "".join(f"{lvl:>10s}" for lvl in LEVELS) + "     of reachable")
    full = {lvl: bench(lvl, (), "doomguy") for lvl in LEVELS}
    print(
        "  everything     "
        + "".join(f"{full[lvl]:10.2f}" for lvl in LEVELS)
        + f"   {sum(full.values()) / sum(CEILING.values()):13.0%}"
    )
    for tag, ablate in arms:
        got = {lvl: bench(lvl, ablate, tag) for lvl in LEVELS}
        share = sum(got.values()) / sum(CEILING.values())
        print(
            f"  {tag:14s} " + "".join(f"{got[lvl]:10.2f}" for lvl in LEVELS) + f"   {share:13.0%}",
            flush=True,
        )


def resolution() -> None:
    """How small a difference this benchmark can actually read, per level.

    Every ablation above is a difference between two means, and a mean of n
    noisy episodes carries an error of its own: std / sqrt(n). Doubling the
    seeds moved the *unchanged* floor on `doom6` from 27.15 to 20.80, which is
    the same thing said the expensive way.

    Printed as a 95 % interval, so it can be read as: differences smaller than
    this are not differences.
    """
    import json
    import math

    print("\nwhat this benchmark can resolve\n")
    print("  level      episodes   mean      std    error of the mean   readable difference")
    for level in LEVELS:
        found = sorted((OUT / level).glob("doomguy.json")) + sorted(
            (OUT / f"wide_{level}").glob("full48.json")
        )
        for path in found:
            data = json.loads(path.read_text())
            rewards = [r["reward"] for r in data["rows"]]
            n = len(rewards)
            mean = sum(rewards) / n
            std = (sum((r - mean) ** 2 for r in rewards) / n) ** 0.5
            sem = std / math.sqrt(n)
            print(
                f"  {level:8s}   {n:6d}  {mean:7.2f}  {std:7.2f}   {sem:13.2f}   {1.96 * sem:17.1f}"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leave-one-out", action="store_true")
    ap.add_argument("--only-one", action="store_true")
    ap.add_argument("--resolution", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.leave_one_out:
        table([(f"no {r}", (r,)) for r in REFLEXES], "what removing one reflex costs")
    if args.only_one:
        arms = [(f"{r} alone", tuple(x for x in REFLEXES if x != r)) for r in REFLEXES]
        table(arms, "what one reflex scores alone")
    if args.resolution:
        resolution()
    return 0


if __name__ == "__main__":
    sys.exit(main())
