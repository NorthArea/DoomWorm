"""Command-line entry point.

doomworm train                     evolve weights (stage 4 small network)
doomworm train --scenario worm     evolve the connectome weights (stage 10)
doomworm compare                   real vs random vs shuffled vs free topology (stage 11)
doomworm benchmark --brain X       run any saved brain through the benchmark (stage 18)
doomworm play --brain brain.json   replay a saved brain on a seeded world
doomworm stimulate ASHL            stimulate connectome neurons, show propagation
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from doomworm import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    from doomworm.experiments.compare_topologies import add_compare_args
    from doomworm.experiments.evolve_small import add_train_args
    from doomworm.experiments.evolve_worm import add_worm_args

    parser = argparse.ArgumentParser(prog="doomworm", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", title="commands")

    train = sub.add_parser("train", help="evolve synaptic weights")
    add_train_args(train)
    train.add_argument("--scenario", choices=["small_food", "worm"], default="small_food")
    add_worm_args(train)

    play = sub.add_parser("play", help="run a saved brain and show the trajectory")
    play.add_argument("--brain", type=Path, required=True)
    play.add_argument("--seed", type=int, default=1000, help="world seed")
    play.add_argument("--steps", type=int, default=600)
    play.add_argument("--every", type=int, default=50)
    play.add_argument("--plot", action="store_true", help="save runs/play_<seed>.png")
    play.add_argument(
        "--maps", choices=["fixed", "random", "apartment"], default=None, help="override brain"
    )
    play.add_argument(
        "--task", choices=["food", "target", "clean"], default=None, help="override brain"
    )
    play.add_argument("--dangers", type=int, default=None, help="override brain")
    play.add_argument(
        "--sensors", choices=["ideal", "vacuum", "noisy"], default=None, help="override brain"
    )

    cmp = sub.add_parser("compare", help="train and compare connectome topologies")
    add_compare_args(cmp)

    bench = sub.add_parser("benchmark", help="run a saved brain through the benchmark")
    bench.add_argument("--brain", type=Path, default=None, help="worm brain JSON")
    bench.add_argument(
        "--scripted",
        choices=["follower", "roomba"],
        default=None,
        help="benchmark a hand-written brain instead of --brain (follower = test driver)",
    )
    bench.add_argument("--name", default=None, help="row name (default: file stem)")
    bench.add_argument("--maps", choices=["fixed", "random", "apartment"], default="apartment")
    bench.add_argument("--task", choices=["food", "target", "clean"], default="clean")
    bench.add_argument("--dangers", type=int, default=0)
    bench.add_argument("--sensors", choices=["ideal", "vacuum", "noisy"], default="vacuum")
    bench.add_argument("--test-seeds", type=int, default=6, help="maps 3000..3000+N-1")
    bench.add_argument("--repeats", type=int, default=3)
    bench.add_argument("--steps", type=int, default=800)
    bench.add_argument("--out-dir", type=Path, default=Path("runs") / "benchmark")
    bench.add_argument(
        "--planner",
        choices=["none", "coverage", "needs"],
        default="none",
        help="wrap with the map layer",
    )

    stim = sub.add_parser("stimulate", help="stimulate neurons of the C. elegans connectome")
    stim.add_argument("neurons", nargs="+", help="neuron names, e.g. ASHL ASHR")
    stim.add_argument("--current", type=float, default=1.0)
    stim.add_argument("--hold", type=int, default=10, help="ticks with stimulus on")
    stim.add_argument("--settle", type=int, default=20, help="ticks after stimulus off")
    stim.add_argument("--gain", type=float, default=0.45)
    stim.add_argument("--decay", type=float, default=0.5)
    stim.add_argument("--top", type=int, default=15)
    stim.add_argument("--threshold", type=float, default=0.01, help="activity counted as active")
    stim.add_argument("--plot", action="store_true", help="save raster + subgraph PNGs to runs/")
    return parser


def run_benchmark_cli(args: argparse.Namespace) -> int:
    """Benchmark a saved brain (worm JSON) and update the leaderboard."""
    from doomworm.brains import WormBrain
    from doomworm.episode import BrainLike
    from doomworm.learning import BenchmarkConfig, run_benchmark, save_result, write_leaderboard

    cfg = BenchmarkConfig(
        maps=args.maps,
        task=args.task,
        dangers=args.dangers,
        sensors=args.sensors,
        test_seeds=tuple(range(3000, 3000 + args.test_seeds)),
        steps=args.steps,
        repeats=args.repeats,
    )
    brain: BrainLike
    if args.scripted == "follower":
        from doomworm.brains import GradientFollower

        brain = GradientFollower()
        name = args.name or "driver_follower"
    elif args.scripted == "roomba":
        from doomworm.brains import RoombaBrain

        brain = RoombaBrain()
        name = args.name or "roomba"
    elif args.brain is not None:
        brain = WormBrain.from_file(
            args.brain, maps=args.maps, task=args.task, dangers=args.dangers
        )
        name = args.name or args.brain.stem
    else:
        raise SystemExit("benchmark: give --brain <file> or --scripted <name>")
    if args.planner != "none":
        from doomworm.brains import PlannerLayer
        from doomworm.environments.sensors import PRESETS

        brain = PlannerLayer(brain, PRESETS[args.sensors], mode=args.planner)
        # "+planner" is the stage-19 coverage row name; other modes carry their own name.
        name += "+planner" if args.planner == "coverage" else f"+{args.planner}"
    print(f"benchmark {name}: {cfg.episodes} episodes on {cfg.maps}/{cfg.task}/{cfg.sensors}")
    result = run_benchmark(
        brain,
        name,
        cfg,
        on_episode=lambda r: print(
            f"  seed {r.seed} rep {r.repeat}: reward {r.reward:7.1f}  coverage {r.coverage:.2f}  "
            f"collisions {r.collisions:3d}  ticks {r.ticks}",
            flush=True,
        ),
    )
    path = save_result(result, args.out_dir)
    print(f"\nsaved {path}\n")
    print(write_leaderboard(args.out_dir))
    return 0


def run_stimulate(args: argparse.Namespace) -> int:
    """Stimulate named neurons and print how activity spreads (Plan §12)."""
    from doomworm.brain import stimulate
    from doomworm.connectome import build_network, load_cook2019
    from doomworm.visualization import plot_active_subgraph, plot_raster, propagation_tree

    worm = load_cook2019()
    unknown = [n for n in args.neurons if n not in worm]
    if unknown:
        raise SystemExit(f"unknown neurons: {unknown}")
    net = build_network(worm, gain=args.gain, decay=args.decay)
    trace = stimulate(net, dict.fromkeys(args.neurons, args.current), args.hold, args.settle)

    first = trace.first_active(args.threshold)
    print(f"stimulus {trace.stimulus} for {args.hold} ticks, gain {args.gain}, decay {args.decay}")
    print(f"neurons active (> {args.threshold}) at some point: {len(first)} of {len(net.neurons)}")
    for name in args.neurons:
        print()
        print(propagation_tree(worm, trace, name, args.threshold))
    print(f"\ntop {args.top} responders:")
    print(f"{'neuron':<8}{'type':<13}{'first':>6}{'peak':>8}")
    for nid, peak in trace.top(args.top):
        print(f"{nid:<8}{worm.neuron(nid).type:<13}{first.get(nid, '-'):>6}{peak:>8.3f}")
    silent = [t for t in range(trace.hold, trace.ticks) if trace.max_activity(t) < args.threshold]
    state = f"silent from tick {silent[0]}" if silent else "still active"
    print(f"\nafter stimulus off: {state}")

    if args.plot:
        stem = "_".join(args.neurons)
        raster = Path("runs") / f"stimulate_{stem}_raster.png"
        graph = Path("runs") / f"stimulate_{stem}_graph.png"
        plot_raster(trace, raster)
        plot_active_subgraph(worm, trace, graph)
        print(f"saved {raster} and {graph}")
    return 0


def run_play(args: argparse.Namespace) -> int:
    """Load a brain JSON and run one episode on the scenario named in its metadata."""
    from doomworm.brain import Simulator, load_brain
    from doomworm.experiments.episode import print_trace, render_ascii, run_episode, save_plot
    from doomworm.experiments.evolve_small import SmallFoodScenario
    from doomworm.experiments.worm_agent import WormScenario
    from doomworm.learning import RewardTracker, Scenario

    net, meta = load_brain(args.brain)
    scenario_name = meta.get("scenario", "small_food")
    scenario: Scenario
    if scenario_name == "small_food":
        scenario = SmallFoodScenario()
    elif scenario_name == "worm":
        params = dict(meta.get("params", {}))
        if args.maps is not None:
            params["maps"] = args.maps
        if args.task is not None:
            params["task"] = args.task
        if args.dangers is not None:
            params["dangers"] = args.dangers
        if args.sensors is not None:
            params["sensors"] = args.sensors
        scenario = WormScenario(**params)
    else:
        raise SystemExit(f"unknown scenario {scenario_name!r}")
    world = scenario.make_world(args.seed)
    tracker = RewardTracker()
    trace = run_episode(
        world,
        Simulator(net),
        scenario.sensory,
        scenario.motor,
        args.steps,
        tracker,
        brain_steps=scenario.brain_steps,
        sensors=scenario.make_sensors(args.seed),
    )

    from doomworm.experiments.worm_agent import summarise

    print_trace(trace, args.every)
    print()
    print(render_ascii(world, trace))
    summary = summarise(trace, world)
    cells = [f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in summary.items()]
    print(f"\nseed {args.seed}: " + "  ".join(cells))
    print(f"reward breakdown {tracker.breakdown}")
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
        if args.scenario == "worm":
            from doomworm.experiments.evolve_worm import run_train as run_train_worm

            return run_train_worm(args)
        from doomworm.experiments.evolve_small import run_train

        return run_train(args)
    if args.command == "stimulate":
        return run_stimulate(args)
    if args.command == "compare":
        from doomworm.experiments.compare_topologies import run_compare

        return run_compare(args)
    if args.command == "benchmark":
        return run_benchmark_cli(args)
    return run_play(args)
