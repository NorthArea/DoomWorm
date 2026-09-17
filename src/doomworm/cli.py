"""Command-line entry point.

doomworm train                     evolve weights (stage 4 small network)
doomworm train --scenario worm     evolve the connectome weights (stage 10)
doomworm compare                   real vs random vs shuffled vs free topology (stage 11)
doomworm benchmark --brain X       run any saved brain through the benchmark (stage 18)
doomworm benchmark --maps doom4 --task doom --scripted doomguy   mini-Doom level (track B)
doomworm evolve --candidate worm   train a bake-off candidate on the benchmark world (stage 21)
doomworm ppo                       train the PPO candidate (optional rl group, stage 21.4)
doomworm play --brain brain.json   replay a saved brain on a seeded world
doomworm drive --teleop            drive the simulator or the machine, record sensors (stage 22)
doomworm compare-log --log X       replay a drive log in the simulator, compare channels
doomworm robustness --brain X      sweep the sensor preset's assumed numbers, table of the drops
doomworm selftest --link tcp       day-one check of the machine: protocol, sensors, wheels
doomworm calibrate --link tcp      measure metres per unit and the wheel base
doomworm plot-log --log X          picture of a drive log (path, rays, wheels)
doomworm stimulate ASHL            stimulate connectome neurons, show propagation
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from doomworm import __version__

SENSOR_PRESETS = ("ideal", "vacuum", "noisy", "car")
MAP_CHOICES = (
    "fixed", "random", "apartment",
    *(f"doom{n}" for n in range(1, 7)),  # mini-Doom on the simulator (track B)
    *(f"vizdoom{n}" for n in range(1, 7)),  # the same levels in the Doom engine (doom group)
    "stock_defend", "stock_corridor", "stock_home",  # stock ViZDoom scenarios (B11)
)  # fmt: skip
TASK_CHOICES = ("food", "target", "clean", "doom")
SCRIPTED = ("follower", "roomba", "doomguy")


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
    play.add_argument("--maps", choices=MAP_CHOICES, default=None, help="override brain")
    play.add_argument("--task", choices=TASK_CHOICES, default=None, help="override brain")
    play.add_argument("--dangers", type=int, default=None, help="override brain")
    play.add_argument("--sensors", choices=SENSOR_PRESETS, default=None, help="override brain")
    play.add_argument(
        "--watch",
        action="store_true",
        help="on a vizdoom level: open the Doom window and run at 35 tics a second",
    )

    cmp = sub.add_parser("compare", help="train and compare connectome topologies")
    add_compare_args(cmp)

    bench = sub.add_parser("benchmark", help="run a saved brain through the benchmark")
    bench.add_argument("--brain", type=Path, default=None, help="worm brain JSON")
    bench.add_argument(
        "--scripted",
        choices=SCRIPTED,
        default=None,
        help="benchmark a hand-written brain instead of --brain (follower = test driver)",
    )
    bench.add_argument("--name", default=None, help="row name (default: file stem)")
    bench.add_argument("--maps", choices=MAP_CHOICES, default="apartment")
    bench.add_argument("--task", choices=TASK_CHOICES, default="clean")
    bench.add_argument("--dangers", type=int, default=0)
    bench.add_argument("--sensors", choices=SENSOR_PRESETS, default="vacuum")
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
    bench.add_argument(
        "--watch",
        action="store_true",
        help="on a vizdoom level: open the Doom window and run at 35 tics a second",
    )

    ev = sub.add_parser("evolve", help="train a bake-off candidate on the benchmark world")
    ev.add_argument(
        "--candidate",
        required=True,
        help="worm, worm_random, worm_shuffled, worm_dense, rnn, ncp (rl group)",
    )
    ev.add_argument("--init-brain", type=Path, default=None, help="start from a saved brain")
    ev.add_argument("--variant-seed", type=int, default=0, help="seed of a control topology")
    ev.add_argument("--layer", choices=["none", "coverage", "needs"], default="needs")
    ev.add_argument("--maps", choices=MAP_CHOICES, default="apartment")
    ev.add_argument("--task", choices=TASK_CHOICES, default="clean")
    ev.add_argument("--sensors", choices=SENSOR_PRESETS, default="vacuum")
    ev.add_argument("--train-seeds", type=int, default=3, help="maps 100..100+N-1")
    ev.add_argument("--train-repeats", type=int, default=1, help="noise seeds per training map")
    ev.add_argument("--steps", type=int, default=800)
    ev.add_argument("--population", type=int, default=40)
    ev.add_argument("--generations", type=int, default=25)
    ev.add_argument("--sigma", type=float, default=0.02)
    ev.add_argument("--init-sigma", type=float, default=None)
    ev.add_argument("--seed", type=int, default=0, help="evolution seed")
    ev.add_argument("--workers", type=int, default=1, help="processes for fitness evaluation")
    ev.add_argument("--out", type=Path, default=None, help="default runs/a2/<candidate>.json")

    ppo = sub.add_parser("ppo", help="train the PPO candidate (needs the rl group)")
    ppo.add_argument("--layer", choices=["none", "coverage", "needs"], default="needs")
    ppo.add_argument("--maps", choices=MAP_CHOICES, default="apartment")
    ppo.add_argument("--task", choices=TASK_CHOICES, default="clean")
    ppo.add_argument("--sensors", choices=SENSOR_PRESETS, default="vacuum")
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

    drv = sub.add_parser("drive", help="drive the simulator or the machine through one link")
    add_link_args(drv)
    drv.add_argument("--brain", type=Path, default=None, help="saved candidate brain")
    drv.add_argument("--scripted", choices=SCRIPTED, default=None)
    drv.add_argument("--teleop", action="store_true", help="wasd / 'l r' pairs from stdin")
    drv.add_argument("--planner", choices=["none", "coverage", "needs"], default="none")
    drv.add_argument("--steps", type=int, default=800)
    drv.add_argument("--every", type=int, default=25, help="print a line every N ticks")
    drv.add_argument("--record", type=Path, default=None, help="drive log (JSON lines)")

    cmp_log = sub.add_parser("compare-log", help="replay a drive log in the simulator, compare")
    cmp_log.add_argument("--log", type=Path, required=True)
    cmp_log.add_argument(
        "--against", type=Path, default=None, help="another log instead of a replay"
    )
    cmp_log.add_argument("--seed", type=int, default=None, help="override the log's world seed")
    cmp_log.add_argument("--sensors", choices=SENSOR_PRESETS, default=None)
    cmp_log.add_argument("--sensor-seed", type=int, default=None)
    cmp_log.add_argument("--room", type=Path, default=None, help="replay in this room file")
    cmp_log.add_argument("--out", type=Path, default=None, help="write the markdown report here")

    rb = sub.add_parser("robustness", help="sweep the assumed sensor numbers around a brain")
    rb.add_argument("--brain", type=Path, default=None, help="saved candidate brain")
    rb.add_argument("--scripted", choices=SCRIPTED, default=None)
    rb.add_argument("--name", default=None)
    rb.add_argument("--planner", choices=["none", "coverage", "needs"], default="needs")
    rb.add_argument("--sensors", choices=SENSOR_PRESETS, default="car")
    rb.add_argument("--maps", choices=MAP_CHOICES, default="apartment")
    rb.add_argument("--task", choices=TASK_CHOICES, default="clean")
    rb.add_argument("--test-seeds", type=int, default=6, help="maps 3000..3000+N-1")
    rb.add_argument("--repeats", type=int, default=2)
    rb.add_argument("--steps", type=int, default=800)
    rb.add_argument(
        "--params", default=None, help="comma-separated subset of the grid (default: all)"
    )
    rb.add_argument(
        "--out", type=Path, default=None, help="markdown table (default: runs/robustness/<name>.md)"
    )

    st = sub.add_parser("selftest", help="day-one check of the machine behind a link")
    add_link_args(st)
    st.add_argument("--frames", type=int, default=10, help="frames to watch at rest")
    st.add_argument("--out", type=Path, default=None, help="write the report here")

    cal = sub.add_parser("calibrate", help="measure metres per unit and the wheel base")
    add_link_args(cal)
    cal.add_argument("--ticks", type=int, default=20, help="ticks of each run")
    cal.add_argument("--out", type=Path, default=Path("runs") / "calibration.json")

    pl = sub.add_parser("plot-log", help="picture of a drive log")
    pl.add_argument("--log", type=Path, required=True)
    pl.add_argument("--out", type=Path, default=None, help="PNG path (default: next to the log)")
    return parser


def add_link_args(parser: argparse.ArgumentParser) -> None:
    """Arguments shared by every command that opens a robot link."""
    parser.add_argument("--link", choices=["sim", "tcp"], default="sim")
    parser.add_argument("--host", default="192.168.4.1", help="robot address for --link tcp")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--sensors", choices=SENSOR_PRESETS, default="vacuum")
    parser.add_argument(
        "--unit-m", type=float, default=None, help="metres per world unit (default: preset)"
    )
    parser.add_argument(
        "--calibration", type=Path, default=None, help="JSON from doomworm calibrate"
    )
    parser.add_argument("--seed", type=int, default=3000, help="world seed (sim link)")
    parser.add_argument("--maps", choices=MAP_CHOICES, default="apartment")
    parser.add_argument("--room", type=Path, default=None, help="room file instead of --maps")
    parser.add_argument("--task", choices=TASK_CHOICES, default="clean")
    parser.add_argument("--sensor-seed", type=int, default=0)


def open_link(args: argparse.Namespace) -> Any:
    """The link the arguments describe: a seeded simulator world or a TCP machine."""
    from dataclasses import replace

    from doomworm.hardware import Calibration, SimLink, connect_tcp, load_calibration

    if args.link == "tcp":
        cal = (
            load_calibration(args.calibration)
            if args.calibration is not None
            else Calibration.for_preset(args.sensors)
        )
        if args.unit_m is not None:
            cal = replace(cal, unit_m=args.unit_m)
        tcp = connect_tcp(args.host, args.port, cal)
        if args.room is not None:
            from doomworm.hardware import load_room

            tcp.room = load_room(args.room).to_dict()
        return tcp
    from doomworm.environments.worlds import build_world

    maps = f"room:{args.room}" if args.room is not None else args.maps
    world = build_world(args.seed, maps, args.task)
    link = SimLink(world, args.sensors, args.sensor_seed, args.seed, maps, args.task)
    if args.room is not None:
        from doomworm.hardware import load_room

        link.room = load_room(args.room).to_dict()
    return link


def run_robustness_cli(args: argparse.Namespace) -> int:
    """Stage 22.1f: one preset parameter at a time, worse than assumed, table of the drops."""
    from doomworm.environments.sensors import PRESETS, SensorConfig
    from doomworm.episode import BrainLike
    from doomworm.learning import CAR_GRID, BenchmarkConfig, sweep, sweep_table, worst_cells

    inner: BrainLike
    if args.scripted == "follower":
        from doomworm.layer import GradientFollower

        inner = GradientFollower()
        name = args.name or "driver_follower"
    elif args.scripted == "roomba":
        from doomworm.candidates import RoombaBrain

        inner = RoombaBrain()
        name = args.name or "roomba"
    elif args.scripted == "doomguy":
        from doomworm.candidates import DoomguyBrain

        inner = DoomguyBrain()
        name = args.name or "doomguy"
    elif args.brain is not None:
        from doomworm.candidates import load_candidate

        inner = load_candidate(args.brain, maps=args.maps, task=args.task)
        name = args.name or args.brain.stem
    else:
        raise SystemExit("robustness: give --brain <file> or --scripted <name>")

    def factory(cfg: SensorConfig) -> BrainLike:
        if args.planner == "none":
            return inner
        from doomworm.layer import PlannerLayer

        return PlannerLayer(inner, cfg, mode=args.planner)

    grid = dict(CAR_GRID)
    if args.params:
        wanted = [p.strip() for p in args.params.split(",") if p.strip()]
        unknown = [p for p in wanted if p not in grid]
        if unknown:
            raise SystemExit(f"robustness: unknown parameters {unknown}; known: {sorted(grid)}")
        grid = {p: grid[p] for p in wanted}
    bench = BenchmarkConfig(
        maps=args.maps,
        task=args.task,
        sensors=args.sensors,
        test_seeds=tuple(range(3000, 3000 + args.test_seeds)),
        steps=args.steps,
        repeats=args.repeats,
    )
    row_name = name if args.planner == "none" else f"{name}+{args.planner}"
    print(f"robustness of {row_name} around {args.sensors}: {bench.episodes} episodes per cell")
    rows = sweep(factory, row_name, PRESETS[args.sensors], grid, bench)
    worst = ", ".join(f"{r.param}={r.label}" for r in worst_cells(rows))
    text = f"# {row_name} around the {args.sensors} preset\n\n" + sweep_table(rows)
    text += f"\nlargest drops: {worst}\n"
    print(text)
    out = args.out or Path("runs") / "robustness" / f"{row_name}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"saved {out}")
    return 0


def run_selftest_cli(args: argparse.Namespace) -> int:
    """Stage 22.2: protocol, sensors and wheels of the machine behind the link."""
    from doomworm.hardware import selftest

    link = open_link(args)
    try:
        report = selftest(link, rest_frames=args.frames)
    finally:
        link.close()
    text = f"# selftest over {link.name} ({args.sensors})\n\n" + report.markdown()
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    return 0 if report.ok else 1


def run_calibrate_cli(args: argparse.Namespace) -> int:
    """Stage 22.2: two measured runs -> runs/calibration.json."""
    from doomworm.hardware import Calibration, calibrate, save_calibration

    link = open_link(args)
    base = getattr(link, "calibration", None) or Calibration.for_preset(args.sensors)
    try:
        measured = calibrate(link, base, ticks=args.ticks)
    finally:
        link.close()
    print(f"saved {save_calibration(measured, args.out)}")
    return 0


def run_plot_log_cli(args: argparse.Namespace) -> int:
    """Stage 22.2: PNG of a drive log."""
    from doomworm.hardware import plot_drive_log, read_drive_log

    meta, rows = read_drive_log(args.log)
    if not rows:
        raise SystemExit(f"plot-log: {args.log} has no rows")
    out = args.out if args.out is not None else args.log.with_suffix(".png")
    print(f"saved {plot_drive_log(rows, out, meta, meta.get('room'))}")
    return 0


def run_drive_cli(args: argparse.Namespace) -> int:
    """Stage 22: one loop for the simulator and the machine, with recording."""
    import sys

    from doomworm.episode import BrainLike
    from doomworm.hardware import Teleop, drive, raw_keys
    from doomworm.hardware.link import RobotLink

    controller: BrainLike
    if args.teleop:
        stream = raw_keys() if sys.stdin.isatty() else sys.stdin
        controller = Teleop(stream)
    elif args.scripted == "follower":
        from doomworm.layer import GradientFollower

        controller = GradientFollower()
    elif args.scripted == "roomba":
        from doomworm.candidates import RoombaBrain

        controller = RoombaBrain()
    elif args.scripted == "doomguy":
        from doomworm.candidates import DoomguyBrain

        controller = DoomguyBrain()
    elif args.brain is not None:
        from doomworm.candidates import load_candidate

        controller = load_candidate(args.brain, maps=args.maps, task=args.task)
    else:
        raise SystemExit("drive: give --teleop, --brain <file> or --scripted <name>")
    if args.planner != "none":
        from doomworm.environments.sensors import PRESETS
        from doomworm.layer import PlannerLayer

        controller = PlannerLayer(controller, PRESETS[args.sensors], mode=args.planner)

    link: RobotLink = open_link(args)
    name = getattr(controller, "name", "?")
    print(f"drive {name} over {link.name} ({args.sensors}), {args.steps} ticks", flush=True)

    def show(row: object) -> None:
        from doomworm.hardware import DriveRow

        assert isinstance(row, DriveRow)
        if row.tick % args.every:
            return
        ch = row.channels
        pose = "" if row.truth is None else f"  true ({row.truth[0]:.1f}, {row.truth[1]:.1f})"
        print(
            f"  t={row.tick:4d} wheels ({row.wheels[0]:+.2f}, {row.wheels[1]:+.2f})  "
            f"front {ch.get('sensor_front', 0.0):.2f}  bump {ch.get('bumper_left', 0.0):.0f}"
            f"{ch.get('bumper_right', 0.0):.0f}  odom ({ch.get('odom_x', 0.0):.1f}, "
            f"{ch.get('odom_y', 0.0):.1f})  battery {ch.get('battery', 0.0):.2f}{pose}",
            flush=True,
        )

    try:
        rows = drive(link, controller, args.steps, log=args.record, on_tick=show)
    finally:
        link.close()
        stream_close = getattr(getattr(controller, "stream", None), "close", None)
        if args.teleop and sys.stdin.isatty() and stream_close is not None:
            stream_close()
    print(f"{len(rows)} ticks driven" + (f", log {args.record}" if args.record else ""))
    return 0


def run_compare_log_cli(args: argparse.Namespace) -> int:
    """Stage 22: replay a drive log in the simulator (or against another log) and report."""
    from dataclasses import replace

    from doomworm.hardware import Calibration, compare_logs, read_drive_log, replay_in_sim

    meta, rows = read_drive_log(args.log)
    if not rows:
        raise SystemExit(f"compare-log: {args.log} has no rows")
    if args.against is not None:
        other_meta, other = read_drive_log(args.against)
        label = f"{args.log} vs {args.against}"
    else:
        other_meta = dict(meta)
        for key in ("seed", "sensors", "sensor_seed"):
            value = getattr(args, key.replace("-", "_"))
            if value is not None:
                other_meta[key] = value
        if args.room is not None:
            from doomworm.hardware import load_room

            other_meta["room"] = load_room(args.room).to_dict()
        other = replay_in_sim(other_meta, rows)
        label = f"{args.log} vs replay ({other_meta.get('sensors')}, seed {other_meta.get('seed')})"
    report = compare_logs(rows, other)
    sensors = str(meta.get("sensors") or "vacuum")
    cal = Calibration.for_preset(sensors)
    if "unit_m" in (meta.get("calibration") or {}):
        cal = replace(cal, unit_m=float(meta["calibration"]["unit_m"]))
    text = f"# {label}\n\n" + report.markdown(cal)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"saved {args.out}")
    return 0


def enable_watch(maps: str | None) -> None:
    """``--watch``: draw the episode in the Doom window instead of running it headless."""
    if maps is None or not maps.startswith("vizdoom"):
        raise SystemExit("--watch needs a Doom-engine level: --maps vizdoom1..vizdoom6")
    try:
        from doomworm.environments.doom.vizdoom_world import set_watch
    except ImportError as e:  # pragma: no cover - depends on the optional group
        raise SystemExit("--watch needs the doom group: uv sync --group doom") from e
    set_watch(True)


def run_benchmark_cli(args: argparse.Namespace) -> int:
    """Benchmark a saved brain (worm JSON) and update the leaderboard."""
    from doomworm.episode import BrainLike
    from doomworm.learning import BenchmarkConfig, run_benchmark, save_result, write_leaderboard

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
        from doomworm.layer import GradientFollower

        brain = GradientFollower()
        name = args.name or "driver_follower"
    elif args.scripted == "roomba":
        from doomworm.candidates import RoombaBrain

        brain = RoombaBrain()
        name = args.name or "roomba"
    elif args.scripted == "doomguy":
        from doomworm.candidates import DoomguyBrain

        brain = DoomguyBrain()
        name = args.name or "doomguy"
    elif args.brain is not None:
        from doomworm.candidates import load_candidate

        brain = load_candidate(args.brain, maps=args.maps, task=args.task, dangers=args.dangers)
        name = args.name or args.brain.stem
    else:
        raise SystemExit("benchmark: give --brain <file> or --scripted <name>")
    if args.planner != "none":
        from doomworm.environments.sensors import PRESETS
        from doomworm.layer import PlannerLayer

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


def run_evolve_cli(args: argparse.Namespace) -> int:
    """Train one candidate with the shared harness (stage 21.2)."""
    from doomworm.candidates import CandidateSpec
    from doomworm.learning import TrainConfig, train_candidate

    spec = CandidateSpec(
        kind=args.candidate,
        init=None if args.init_brain is None else str(args.init_brain),
        variant_seed=args.variant_seed,
        maps=args.maps,
        task=args.task,
        sensors=args.sensors,
    )
    cfg = TrainConfig(
        layer=args.layer,
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
    )
    out = args.out or Path("runs") / "a2" / f"{spec.label()}.json"
    print(
        f"evolve {spec.label()} under layer {cfg.layer}: {cfg.population} x {cfg.generations} "
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
    print(f"benchmark: doomworm benchmark --brain {out} --planner {cfg.layer}")
    return 0


def run_ppo_cli(args: argparse.Namespace) -> int:
    """Train the PPO candidate (stage 21.4)."""
    from doomworm.learning.rl import PPOConfig, train_ppo

    cfg = PPOConfig(
        maps=args.maps,
        task=args.task,
        sensors=args.sensors,
        steps=args.steps,
        layer=args.layer,
        train_seeds=tuple(range(100, 100 + args.train_seeds)),
        timesteps=args.timesteps,
        n_steps=args.n_steps,
        seed=args.seed,
    )
    print(f"ppo under layer {cfg.layer}: {cfg.timesteps} steps on {cfg.train_seeds} -> {args.out}")
    out = train_ppo(cfg, args.out, verbose=1)
    print(f"saved {out}; benchmark: doomworm benchmark --brain {out} --planner {cfg.layer}")
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
    from doomworm.episode import Record, run_brain_episode
    from doomworm.experiments.episode import print_trace, render_ascii, run_episode, save_plot
    from doomworm.experiments.evolve_small import SmallFoodScenario
    from doomworm.experiments.worm_agent import WormScenario
    from doomworm.learning import RewardTracker, Scenario

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
        from doomworm.candidates import WormBrain

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
    if args.command == "evolve":
        return run_evolve_cli(args)
    if args.command == "ppo":
        return run_ppo_cli(args)
    if args.command == "drive":
        return run_drive_cli(args)
    if args.command == "compare-log":
        return run_compare_log_cli(args)
    if args.command == "selftest":
        return run_selftest_cli(args)
    if args.command == "robustness":
        return run_robustness_cli(args)
    if args.command == "calibrate":
        return run_calibrate_cli(args)
    if args.command == "plot-log":
        return run_plot_log_cli(args)
    return run_play(args)
