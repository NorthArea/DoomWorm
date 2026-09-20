"""Command-line entry point.

doomworm train                     evolve weights (stage 4 small network)
doomworm train --scenario worm     evolve the connectome weights (stage 10)
doomworm compare                   real vs random vs shuffled vs free topology (stage 11)
doomworm benchmark --brain X       run any saved brain through the benchmark (stage 18)
doomworm benchmark --maps doom4 --task doom --scripted doomguy   mini-Doom level (track B)
doomworm evolve --candidate worm   train a bake-off candidate on the benchmark world (stage 21)
doomworm ppo                       train the PPO candidate (optional rl group, stage 21.4)
doomworm play --brain brain.json   replay a saved brain on a seeded world
doomworm stimulate ASHL            stimulate connectome neurons, show propagation
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from wormlab import __version__

SENSOR_PRESETS = ("ideal", "vacuum", "noisy", "car")
MAP_CHOICES = (
    "fixed", "random", "apartment",
    *(f"doom{n}" for n in range(1, 7)),  # mini-Doom on the simulator (track B)
    *(f"vizdoom{n}" for n in range(1, 7)),  # the same levels in the Doom engine (doom group)
    "stock_defend", "stock_corridor", "stock_home",  # stock ViZDoom scenarios (B11)
    *(f"e{e}m{m}" for e in range(1, 5) for m in range(1, 10)),  # the classic episode maps
)  # fmt: skip
TASK_CHOICES = ("food", "target", "doom")
SCRIPTED = ("follower", "doomguy", "flailing")


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    from wormlab.experiments.compare_topologies import add_compare_args
    from wormlab.experiments.evolve_small import add_train_args
    from wormlab.experiments.evolve_worm import add_worm_args

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
    play.add_argument("--maps", choices=MAP_CHOICES, default=None, help="override brain")
    play.add_argument("--task", choices=TASK_CHOICES, default=None, help="override brain")
    play.add_argument("--dangers", type=int, default=None, help="override brain")
    play.add_argument("--sensors", choices=SENSOR_PRESETS, default=None, help="override brain")
    play.add_argument(
        "--watch",
        action="store_true",
        help="on an engine level (vizdoom*, stock_*): open the Doom window, 35 tics a second",
    )

    cmp = sub.add_parser("compare", help="train and compare connectome topologies")
    add_compare_args(cmp)

    bench = sub.add_parser("benchmark", help="run a saved brain through the benchmark")
    bench.add_argument("--brain", type=Path, default=None, help="worm brain JSON")
    bench.add_argument(
        "--ablate",
        nargs="*",
        default=(),
        metavar="REFLEX",
        help="doomguy only: reflexes to switch off (stage 26)",
    )
    bench.add_argument(
        "--scripted",
        choices=SCRIPTED,
        default=None,
        help="benchmark a hand-written brain instead of --brain (follower = test driver)",
    )
    bench.add_argument("--name", default=None, help="row name (default: file stem)")
    bench.add_argument(
        "--memory",
        action="store_true",
        help="wrap the brain in the memory layer (the condition it was trained under)",
    )
    bench.add_argument("--maps", choices=MAP_CHOICES, default="doom4")
    bench.add_argument("--task", choices=TASK_CHOICES, default="doom")
    bench.add_argument("--dangers", type=int, default=0)
    bench.add_argument("--sensors", choices=SENSOR_PRESETS, default="ideal")
    bench.add_argument("--test-seeds", type=int, default=6, help="maps 3000..3000+N-1")
    bench.add_argument("--repeats", type=int, default=3)
    bench.add_argument("--steps", type=int, default=800)
    bench.add_argument("--out-dir", type=Path, default=Path("runs") / "benchmark")
    bench.add_argument(
        "--watch",
        action="store_true",
        help="on an engine level (vizdoom*, stock_*): open the Doom window, 35 tics a second",
    )

    ev = sub.add_parser("evolve", help="train a bake-off candidate on the benchmark world")
    ev.add_argument(
        "--candidate",
        required=True,
        help="worm, worm_random, worm_shuffled, worm_dense, rnn, ncp (rl group)",
    )
    ev.add_argument("--init-brain", type=Path, default=None, help="start from a saved brain")
    ev.add_argument(
        "--memory",
        action="store_true",
        help="train under the memory layer: the brain also senses unvisited ground",
    )
    ev.add_argument("--variant-seed", type=int, default=0, help="seed of a control topology")
    ev.add_argument(
        "--trainable",
        choices=["all", "interface", "sensory"],
        default="all",
        help="which synapses the search may move (stage 18): 5905, 1819 or 506",
    )
    ev.add_argument(
        "--tune-gains",
        action="store_true",
        help="put the adapter gains in the genome instead of fixing them by hand",
    )
    ev.add_argument("--maps", choices=MAP_CHOICES, default="doom4")
    ev.add_argument("--task", choices=TASK_CHOICES, default="doom")
    ev.add_argument("--sensors", choices=SENSOR_PRESETS, default="ideal")
    ev.add_argument("--train-seeds", type=int, default=3, help="maps 100..100+N-1")
    ev.add_argument("--train-repeats", type=int, default=1, help="noise seeds per training map")
    ev.add_argument("--steps", type=int, default=800)
    ev.add_argument("--population", type=int, default=40)
    ev.add_argument("--generations", type=int, default=25)
    ev.add_argument("--sigma", type=float, default=0.02)
    ev.add_argument(
        "--search",
        choices=["mutation", "cma"],
        default="mutation",
        help="fixed-sigma mutation (stage 4) or sep-CMA-ES (stage 18)",
    )
    ev.add_argument("--init-sigma", type=float, default=None)
    ev.add_argument("--seed", type=int, default=0, help="evolution seed")
    ev.add_argument("--workers", type=int, default=1, help="processes for fitness evaluation")
    ev.add_argument("--out", type=Path, default=None, help="default runs/a2/<candidate>.json")

    ppo = sub.add_parser("ppo", help="train the PPO candidate (needs the rl group)")
    ppo.add_argument("--maps", choices=MAP_CHOICES, default="doom4")
    ppo.add_argument("--task", choices=TASK_CHOICES, default="doom")
    ppo.add_argument("--sensors", choices=SENSOR_PRESETS, default="ideal")
    ppo.add_argument("--train-seeds", type=int, default=3, help="maps 100..100+N-1")
    ppo.add_argument("--steps", type=int, default=800)
    ppo.add_argument("--timesteps", type=int, default=300_000)
    ppo.add_argument("--n-steps", type=int, default=2048)
    ppo.add_argument("--seed", type=int, default=0)
    ppo.add_argument("--out", type=Path, default=Path("runs") / "a2" / "ppo.json")

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


def enable_watch(maps: str | None) -> None:
    """``--watch``: draw the episode in the Doom window instead of running it headless."""
    engine = maps is not None and (maps.startswith("vizdoom") or maps.startswith("stock_"))
    if not engine:  # the simulator levels have no window: the engine is what draws
        raise SystemExit("--watch needs a Doom-engine level: --maps vizdoom1..vizdoom6 or stock_*")
    try:
        from doomworm.engine import set_watch
    except ImportError as e:  # pragma: no cover - depends on the optional group
        raise SystemExit("--watch needs the doom group: uv sync --group doom") from e
    set_watch(True)


def run_benchmark_cli(args: argparse.Namespace) -> int:
    """Benchmark a saved brain (worm JSON) and update the leaderboard."""
    from wormlab.episode import BrainLike
    from wormlab.learning import BenchmarkConfig, run_benchmark, save_result, write_leaderboard

    if args.watch:
        enable_watch(args.maps)
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
        from wormlab.candidates import GradientFollower

        brain = GradientFollower()
        name = args.name or "driver_follower"
    elif args.scripted == "flailing":
        # Stage 25: the floor under the floor. Uniform intents, no relation to
        # the channels -- what a row scores before any behaviour is involved.
        from wormlab.learning.search import RandomProposer

        brain = RandomProposer(seed=0)  # one stream across the episodes, reset() keeps it
        name = args.name or "flailing"
    elif args.scripted == "doomguy":
        from doomworm import DoomguyBrain

        # stage 26: --ablate switches reflexes off, so the cost of removing one
        # is measured down the same path every published row came from
        brain = DoomguyBrain(ablate=args.ablate)
        name = args.name or ("doomguy" if not args.ablate else f"doomguy-{'-'.join(args.ablate)}")
    elif args.brain is not None:
        from wormlab.candidates import load_candidate

        brain = load_candidate(args.brain, maps=args.maps, task=args.task, dangers=args.dangers)
        name = args.name or args.brain.stem
    else:
        raise SystemExit("benchmark: give --brain <file> or --scripted <name>")
    if args.memory:
        from wormlab.layer import MemoryLayer

        brain = MemoryLayer(brain)
        name += "+memory"
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


def run_evolve_cli(args: argparse.Namespace) -> int:
    """Train one candidate with the shared harness (stage 21.2)."""
    from wormlab.candidates import CandidateSpec
    from wormlab.learning import TrainConfig, train_candidate

    spec = CandidateSpec(
        kind=args.candidate,
        init=None if args.init_brain is None else str(args.init_brain),
        variant_seed=args.variant_seed,
        maps=args.maps,
        task=args.task,
        sensors=args.sensors,
        params={"trainable": args.trainable, "tune_gains": args.tune_gains},
    )
    cfg = TrainConfig(
        layer="memory" if args.memory else "none",
        train_seeds=tuple(range(100, 100 + args.train_seeds)),
        train_repeats=args.train_repeats,
        steps=args.steps,
        population=args.population,
        generations=args.generations,
        sigma=args.sigma,
        init_sigma=args.init_sigma,
        mutation_fraction=0.05 if args.candidate == "worm_dense" else 1.0,
        seed=args.seed,
        workers=args.workers,
        search=args.search,
    )
    out = args.out or Path("runs") / "a2" / f"{spec.label()}.json"
    print(
        f"evolve {spec.label()}: {cfg.population} x {cfg.generations} "
        f"on maps {cfg.train_seeds}, {cfg.workers} workers -> {out}"
    )
    result = train_candidate(
        spec,
        cfg,
        out,
        on_generation=lambda g: print(
            f"  gen {g.generation:3d}  best {g.best:8.2f}  mean {g.mean:8.2f}", flush=True
        ),
    )
    print(f"best train fitness {result.best_fitness:.2f}; saved {out}")
    print(f"benchmark: doomworm benchmark --brain {out}")
    return 0


def run_ppo_cli(args: argparse.Namespace) -> int:
    """Train the PPO candidate (stage 21.4)."""
    from wormlab.learning.rl import PPOConfig, train_ppo

    cfg = PPOConfig(
        maps=args.maps,
        task=args.task,
        sensors=args.sensors,
        steps=args.steps,
        train_seeds=tuple(range(100, 100 + args.train_seeds)),
        timesteps=args.timesteps,
        n_steps=args.n_steps,
        seed=args.seed,
    )
    print(f"ppo: {cfg.timesteps} steps on {cfg.train_seeds} -> {args.out}")
    out = train_ppo(cfg, args.out, verbose=1)
    print(f"saved {out}; benchmark: doomworm benchmark --brain {out}")
    return 0


def run_stimulate(args: argparse.Namespace) -> int:
    """Stimulate named neurons and print how activity spreads (Plan §12)."""
    from wormlab.brain import stimulate
    from wormlab.connectome import build_network, load_cook2019
    from wormlab.visualization import plot_active_subgraph, plot_raster, propagation_tree

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
    from wormlab.brain import Simulator, load_brain
    from wormlab.episode import Record, run_brain_episode
    from wormlab.experiments.episode import print_trace, render_ascii, run_episode, save_plot
    from wormlab.experiments.evolve_small import SmallFoodScenario
    from wormlab.experiments.worm_agent import WormScenario
    from wormlab.learning import RewardTracker, Scenario

    net, meta = load_brain(args.brain)
    scenario_name = meta.get("scenario", "small_food")
    scenario: Scenario
    trace: list[Record]
    tracker = RewardTracker()
    if scenario_name == "small_food":
        scenario = SmallFoodScenario()
        world = scenario.make_world(args.seed)
        trace = run_episode(
            world,
            Simulator(net),
            scenario.sensory,
            scenario.motor,
            args.steps,
            tracker,
            brain_steps=scenario.brain_steps,
            sensors=None,  # the small scenario has no sensor suite
        )
    elif scenario_name == "worm":
        from wormlab.candidates import WormBrain

        overrides = {
            k: v
            for k, v in (
                ("maps", args.maps),
                ("task", args.task),
                ("dangers", args.dangers),
                ("sensors", args.sensors),
            )
            if v is not None
        }
        brain = WormBrain.from_file(args.brain, **overrides)
        params = brain.meta["params"]
        if args.watch:
            enable_watch(params.get("maps"))
        scenario = WormScenario(**params)
        world = scenario.make_world(args.seed)
        # the Brain loop: same trace as the old loop, plus the trigger on Doom levels
        trace = run_brain_episode(
            world, brain, args.steps, tracker, sensors=scenario.make_sensors(args.seed)
        )
    else:
        raise SystemExit(f"unknown scenario {scenario_name!r}")

    from wormlab.experiments.worm_agent import summarise

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
            from wormlab.experiments.evolve_worm import run_train as run_train_worm

            return run_train_worm(args)
        from wormlab.experiments.evolve_small import run_train

        return run_train(args)
    if args.command == "stimulate":
        return run_stimulate(args)
    if args.command == "compare":
        from wormlab.experiments.compare_topologies import run_compare

        return run_compare(args)
    if args.command == "benchmark":
        return run_benchmark_cli(args)
    if args.command == "evolve":
        return run_evolve_cli(args)
    if args.command == "ppo":
        return run_ppo_cli(args)
    return run_play(args)


if __name__ == "__main__":  # `python -m wormlab.cli` used to exit 0 in silence
    raise SystemExit(main())
