"""Stage 21: does searching over the worm's proposals beat the worm alone?

    uv run python scripts/search_probe.py

Three rows on the same maps and seeds: the deterministic brain, the same brain
made stochastic (which is what makes a search possible at all), and the search
over its proposals. If the third does not win, there is nothing worth teaching
back in stage 23.
"""

from __future__ import annotations

import sys
from pathlib import Path

from doomworm.candidates import WormBrain
from doomworm.environments.worlds import build_world
from doomworm.episode import run_brain_episode
from doomworm.experiments.worm_agent import WormScenario
from doomworm.learning.reward import RewardConfig, RewardTracker
from doomworm.learning.search import SearchConfig, searched_episode

ROOT = Path(__file__).resolve().parent.parent

BRAIN = ROOT / "runs/squeeze2/seed0/cma_interface.json"  # the best worm the project has


def brain_for(level: str, noise: float) -> WormBrain:
    if BRAIN.exists():
        worm = WormBrain.from_file(BRAIN, maps=level, task="doom", sensors="ideal")
    else:
        worm = WormBrain.from_scenario(
            WormScenario(maps=level, task="doom", sensors="ideal")
        )
    worm.noise = noise
    return worm


def plain(level: str, noise: float, seeds: range, steps: int) -> tuple[float, float]:
    total = exits = 0.0
    for seed in seeds:
        worm = brain_for(level, noise)
        worm.episode = seed
        world = build_world(seed, level, "doom")
        tracker = RewardTracker(RewardConfig())
        run_brain_episode(world, worm, steps, tracker)
        total += sum(tracker.breakdown.values())
        exits += float(world.exited)
    n = len(seeds)
    return total / n, exits / n


def searched(level: str, cfg: SearchConfig, seeds: range, steps: int) -> tuple[float, float]:
    total = exits = 0.0
    for seed in seeds:
        worm = brain_for(level, cfg.noise)
        worm.episode = seed
        world = build_world(seed, level, "doom")
        reward, _rows = searched_episode(world, worm, steps, cfg)
        total += reward
        exits += float(world.exited)
    n = len(seeds)
    return total / n, exits / n


def main() -> int:
    seeds, steps = range(3000, 3012), 600
    for level in ("doom4", "doom2", "doom6"):
        print(f"\n-- {level}: 12 unseen maps, {steps} ticks --", flush=True)
        for label, config in (
            ("brain alone", None),
            ("brain + noise", SearchConfig(0, 0, 0, 0.3)),
            ("search 6 x 12", SearchConfig(6, 12, 5, 0.3)),
            ("search 12 x 20", SearchConfig(12, 20, 5, 0.3)),
        ):
            if config is None:
                reward, exits = plain(level, 0.0, seeds, steps)
            elif config.candidates == 0:
                reward, exits = plain(level, config.noise, seeds, steps)
            else:
                reward, exits = searched(level, config, seeds, steps)
            mark = "   <-- EXITS" if exits > 0 else ""
            print(f"   {label:16s} reward {reward:7.2f}   exits {exits:.2f}{mark}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
