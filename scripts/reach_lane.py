"""Stage 25: the denominator, and the floor under the floor.

    uv run python scripts/reach_lane.py --ceiling     # what the maps pay if you take everything
    uv run python scripts/reach_lane.py --floor       # what a policy with no behaviour scores

Twenty-four rows of findings.md compare trained brains against trained controls
-- a shuffled connectome, a random topology, a net from scratch. Every one of
them is a comparison between things that were optimised. None of them says what
a row scores when *nothing* is optimised, and none of them says what fraction of
the available reward any row is collecting.

So: the ceiling is counted off the map (the exit, the monsters, the floor that
can be walked), which turns every published reward into a fraction. And the
floor is uniform intents with no relation to the channels, benchmarked under
exactly the conditions the published rows used -- 600 ticks, 12 seeds, ideal
sensors -- because a number only means something against what chance gives.

Run with enough episodes. A single lucky kill is worth 12 points here, which is
more than any trained brain's whole episode, so a four-episode mean of a random
policy is mostly one coin flip. That mistake is in docs/knowledge/method.md; the
default below is what it takes for it not to happen.
"""

from __future__ import annotations

import argparse
import statistics as st
import subprocess
import sys
from pathlib import Path

from wormlab.environments.simple_2d import World
from wormlab.environments.worlds import build_world
from wormlab.learning.reward import RewardConfig

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs/reach"
LEVELS = ("doom1", "doom2", "doom4", "doom6")
STEPS, SEEDS, REPEATS = "600", "12", "2"


def ceiling(world: World, cfg: RewardConfig) -> dict[str, float]:
    """What this map pays if everything on it is collected.

    Exploration's bound is every cell the agent could stand in -- clearance above
    its own radius -- at `new_cell` each. A monster pays its kill plus the one
    hit that is needed to land it; more hits only if it takes more.
    """
    cells = sum(
        1
        for x in range(int(world.width / cfg.cell_size))
        for y in range(int(world.height / cfg.cell_size))
        if world.clearance((x + 0.5) * cfg.cell_size, (y + 0.5) * cfg.cell_size)
        > world.agent_radius
    )
    return {
        "exit": cfg.target if world.target is not None else 0.0,
        "monsters": len(world.enemies) * (cfg.kill + cfg.hit),
        "floor": cells * cfg.new_cell,
    }


def travel(world: World, ticks: int) -> float:
    """The furthest the agent can move in `ticks`: its own speed constant times them.

    Driving it forward and measuring gives a smaller number, because it runs
    into a wall -- which is a fact about the map, not about the agent. This is
    the generous bound on purpose: a straight line that never turns and never
    revisits a cell.
    """
    return ticks * world.speed


def show_ceiling(seeds: range, ticks: int) -> None:
    """Two rows per level: what is on the map, and what an episode could reach.

    The second is the one a score should be read against. A reward item has to
    be *arrived at*, and this agent moves 0.2 units a tick: a 600-tick episode
    buys 120 units of travel in a 20x20 room whose floor is 283 cells. Most of the
    exploration term is on the map and out of reach in the time given, and the
    bound below is still generous -- a straight line at full speed that never
    turns and never crosses itself.
    """
    cfg = RewardConfig()
    reach = travel(build_world(seeds[0], LEVELS[0], "doom"), ticks)
    print(f"what the maps pay, averaged over {len(seeds)} seeds each")
    print(f"{ticks} ticks buys {reach:.0f} units of travel\n")
    print("  level     exit  monsters   floor   total     of which reachable")
    for level in LEVELS:
        caps = [ceiling(build_world(s, level, "doom"), cfg) for s in seeds]
        cap = {k: st.fmean(c[k] for c in caps) for k in caps[0]}
        walked = min(cap["floor"], reach * cfg.new_cell / cfg.cell_size)
        print(
            f"  {level:8s} {cap['exit']:5.1f}  {cap['monsters']:8.1f}  {cap['floor']:6.1f}"
            f"  {sum(cap.values()):6.1f}     {cap['exit'] + cap['monsters'] + walked:6.1f}"
        )


def show_floor(who: tuple[str, ...]) -> None:
    """Benchmark scripted brains under the conditions the published rows used."""
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"\n{SEEDS} seeds x {REPEATS} repeats x {STEPS} ticks, ideal sensors\n")
    for level in LEVELS:
        for name in who:
            cmd = [
                "wormlab", "benchmark",
                "--maps", level, "--task", "doom", "--sensors", "ideal",
                "--scripted", name,
                "--steps", STEPS, "--test-seeds", SEEDS, "--repeats", REPEATS,
                "--out-dir", str(OUT / level),
            ]  # fmt: skip
            out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout
            row = [ln for ln in out.splitlines() if ln.startswith("| 1 |")]
            print(f"  {level:8s} {row[0] if row else 'FAILED'}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ceiling", action="store_true")
    ap.add_argument("--floor", action="store_true")
    ap.add_argument("--seeds", type=int, default=12)
    args = ap.parse_args()
    if args.ceiling:
        show_ceiling(range(3000, 3000 + args.seeds), int(STEPS))
    if args.floor:
        show_floor(("flailing", "doomguy"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
