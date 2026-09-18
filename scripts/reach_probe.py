"""Stage 25: how much of the reward is reachable, and from how far away.

    uv run python scripts/reach_probe.py

Every row since stage 21 has said the same thing in a different way: this brain's
answer depends on the state, the dependence is worth about 0.15 in intent units,
and correcting the actions -- by search, by distillation, by a better optimiser --
moves the reward by nothing. Stage 24 closed that line with each stated reason
eliminated in turn, which leaves a question nobody here has measured.

**Is the reward reachable at all by a policy of this shape?**

Two halves, and the second is the one that matters.

    the ceiling   what is physically on the map: the exit, the monsters, the
                  floor that can be explored. Counted from the world, not
                  guessed, so every score in findings.md gets a denominator.

    the reach     a look-ahead of H ticks over the same proposals, swept over H.
                  A reward item appears in the take at the H where a searcher can
                  first *see* it. If kills only arrive at H=48 and every search in
                  this project ran at H=12, then the brain was never being asked
                  the question we thought we were asking.

The random proposer is the control at every horizon, because stage 23 found it
searches as well as the connectome does and stage 24 found it teaches as well.
If the sweep separates them anywhere, that is the first place this wiring has
been worth something at depth.
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

from wormlab.candidates import WormBrain
from wormlab.environments.simple_2d import World
from wormlab.environments.worlds import build_world
from wormlab.learning.reward import RewardConfig, RewardTracker
from wormlab.learning.search import RandomProposer, SearchConfig, searched_episode

ROOT = Path(__file__).resolve().parent.parent
BRAIN = ROOT / "docs/doom/brains/squeeze2/seed0/cma_interface.json"
LEVEL, STEPS, NOISE = "doom4", 300, 0.3
PARTS = ("target", "kill", "hit", "new_cell", "damage", "death", "collision")


def ceiling(world: World, cfg: RewardConfig) -> dict[str, float]:
    """What this map pays if everything that can be collected is collected.

    The floor area is the honest upper bound on exploration: every cell whose
    centre the agent could actually stand in -- `clearance` above its radius --
    at `new_cell` each. Monsters pay a kill plus one hit.
    """
    cells = sum(
        1
        for x in range(int(world.width / cfg.cell_size))
        for y in range(int(world.height / cfg.cell_size))
        if world.clearance((x + 0.5) * cfg.cell_size, (y + 0.5) * cfg.cell_size)
        > world.agent_radius
    )
    monsters = len(world.enemies)
    return {
        "target": cfg.target if world.target is not None else 0.0,
        "kill": monsters * cfg.kill,
        "hit": monsters * cfg.hit,  # one hit each; more only if they take more
        "new_cell": cells * cfg.new_cell,
    }


def play(seed: int, horizon: int, control: bool) -> tuple[dict[str, float], float]:
    """One episode at one look-ahead depth; returns the breakdown and the total."""
    if control:
        brain: object = RandomProposer(seed=seed)
    else:
        brain = WormBrain.from_file(BRAIN, maps=LEVEL, task="doom", sensors="ideal")
        brain.noise, brain.episode = NOISE, seed  # type: ignore[attr-defined]
    world = build_world(seed, LEVEL, "doom")
    if horizon == 0:  # no search: the brain alone, one answer per tick
        from wormlab.episode import run_brain_episode

        tracker = RewardTracker(RewardConfig())
        run_brain_episode(world, brain, STEPS, tracker)  # type: ignore[arg-type]
        return dict(tracker.breakdown), sum(tracker.breakdown.values())
    cfg = SearchConfig(candidates=8, horizon=horizon, plan_every=1, noise=NOISE)
    tracker = RewardTracker(RewardConfig())
    total, _rows = searched_episode(world, brain, STEPS, cfg, tracker.config)  # type: ignore[arg-type]
    return _replay(seed, world), total


def _replay(seed: int, world: World) -> dict[str, float]:
    """What the episode actually collected, read off the world rather than the tracker."""
    del seed
    return {
        "kills": float(world.kills),
        "hits": float(world.hits),
        "exited": float(world.exited),
        "shots": float(world.shots),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--horizons", type=int, nargs="+", default=[0, 6, 12, 24, 48])
    args = ap.parse_args()
    seeds = range(3000, 3000 + args.seeds)
    cfg = RewardConfig()

    caps = [ceiling(build_world(s, LEVEL, "doom"), cfg) for s in seeds]
    cap = {k: st.fmean(c[k] for c in caps) for k in caps[0]}
    print(f"{LEVEL}, {len(seeds)} maps, {STEPS} ticks each")
    print(
        f"  on the map: exit {cap['target']:.1f}, monsters {cap['kill'] / cfg.kill:.1f} "
        f"worth {cap['kill'] + cap['hit']:.1f}, floor worth {cap['new_cell']:.1f} "
        f"-- {sum(cap.values()):.1f} available\n"
    )
    print("  look-ahead        reward   kills   hits  exits   shots")
    for control in (False, True):
        print(f"  {'proposals from nothing' if control else 'the worm proposing'}")
        for h in args.horizons:
            rows = [play(s, h, control) for s in seeds]
            total = st.fmean(t for _b, t in rows)
            got = {
                k: st.fmean(b.get(k, 0.0) for b, _t in rows)
                for k in ("kills", "hits", "exited", "shots")
            }
            depth = "none" if h == 0 else f"{h:d} ticks"
            print(
                f"    {depth:14s}  {total:7.2f}  {got['kills']:5.2f}  {got['hits']:5.2f}"
                f"  {got['exited']:5.2f}  {got['shots']:6.1f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
