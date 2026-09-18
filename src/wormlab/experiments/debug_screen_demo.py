"""Stage 6 demo: the §41 debug screen as a GIF, using the hand-wired stage-2 brain.

Run: ``uv run python -m wormlab.experiments.debug_screen_demo [--steps N] [--every K]``
"""

from __future__ import annotations

import argparse
from pathlib import Path

from wormlab.episode import run_episode
from wormlab.experiments import food_agent
from wormlab.learning import RewardTracker
from wormlab.visualization import save_debug_frame, save_debug_gif


def main(argv: list[str] | None = None) -> int:
    """Run the stage-2 scenario with activity recording and write GIF + PNG."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--every", type=int, default=5)
    args = parser.parse_args(argv)

    world, sim, sensory, motor = food_agent.build_scenario()
    tracker = RewardTracker()
    trace = run_episode(world, sim, sensory, motor, args.steps, tracker, record_activity=True)
    gif = Path("runs") / "stage6_debug.gif"
    png = Path("runs") / "stage6_debug_frame.png"
    frames = save_debug_gif(world, trace, gif, every=args.every)
    save_debug_frame(world, trace, min(len(trace) - 1, 175), png)
    print(f"{len(trace)} ticks, food {world.food_eaten}, collisions {world.collisions}")
    print(f"saved {gif} ({frames} frames) and {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
