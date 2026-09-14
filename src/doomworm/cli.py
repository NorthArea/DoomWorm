"""Command-line entry point.

doomworm train                     evolve weights (stage 4 scenario)
doomworm play --brain brain.json   replay a saved brain on a seeded world
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from doomworm import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    from doomworm.experiments.evolve_small import add_train_args

    parser = argparse.ArgumentParser(prog="doomworm", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", title="commands")

    train = sub.add_parser("train", help="evolve synaptic weights")
    add_train_args(train)

    play = sub.add_parser("play", help="run a saved brain and show the trajectory")
    play.add_argument("--brain", type=Path, required=True)
    play.add_argument("--seed", type=int, default=1000, help="world seed")
    play.add_argument("--steps", type=int, default=600)
    play.add_argument("--every", type=int, default=50)
    play.add_argument("--plot", action="store_true", help="save runs/play_<seed>.png")
    return parser


def run_play(args: argparse.Namespace) -> int:
    """Load a brain JSON and run one episode on the scenario named in its metadata."""
    from doomworm.brain import Simulator, load_brain
    from doomworm.experiments.episode import print_trace, render_ascii, run_episode, save_plot
    from doomworm.experiments.evolve_small import SCENARIO_NAME, SmallFoodScenario
    from doomworm.learning import RewardTracker

    net, meta = load_brain(args.brain)
    scenario_name = meta.get("scenario", SCENARIO_NAME)
    if scenario_name != SCENARIO_NAME:
        raise SystemExit(f"unknown scenario {scenario_name!r}")
    scenario = SmallFoodScenario()
    world = scenario.make_world(args.seed)
    tracker = RewardTracker()
    trace = run_episode(
        world, Simulator(net), scenario.sensory, scenario.motor, args.steps, tracker
    )

    print_trace(trace, args.every)
    print()
    print(render_ascii(world, trace))
    print(
        f"\nseed {args.seed}: {len(trace)} ticks, food {world.food_eaten}, "
        f"collisions {world.collisions}, reward {tracker.total:.1f} {tracker.breakdown}"
    )
    if args.plot:
        out = Path("runs") / f"play_{args.seed}.png"
        save_plot(world, trace, out, f"{args.brain.name} on seed {args.seed}")
        print(f"saved {out}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return an exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "train":
        from doomworm.experiments.evolve_small import run_train

        return run_train(args)
    return run_play(args)
