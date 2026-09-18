"""Stage 24: record the teacher where the student actually lives.

    uv run python scripts/dagger_lane.py --gate       # is the label a function of the state?
    uv run python scripts/dagger_lane.py --collect    # only if the gate says yes
    uv run python scripts/dagger_lane.py --train
    uv run python scripts/dagger_lane.py --benchmark

Stage 23 distilled a search and taught the brain nothing. Stage 23b found the
reason and it was not the wiring: along its own trajectory the worm's answer
varies five times more between states than within one, and along the *search's*
trajectory only 1.3 times. The demonstrations had been recorded in the one
region where this brain has least to say.

That is DAgger's mismatch exactly, so this is DAgger's correction: the
**student** drives, and the search labels what it should have done at the
states the student actually reaches. Nothing else changes.

The gate comes first and is not optional. Labelling is expensive and training
on it is more so, and stage 23 spent both before asking whether the labels
carried any state-dependent structure at all. The same-state-twice ratio in
`docs/knowledge/method.md` answers that in minutes, and this lane refuses to
collect until it has.
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

import numpy as np

from wormlab.body import Drive, drive_of
from wormlab.candidates import WormBrain
from wormlab.environments.worlds import build_world
from wormlab.episode import run_brain_episode
from wormlab.learning.cmaes import CMAConfig, cma_es
from wormlab.learning.distill import (
    Demonstration,
    imitation_error,
    imitation_fitness,
    load_demos,
    save_demos,
)
from wormlab.learning.reward import RewardConfig, RewardTracker
from wormlab.learning.search import RandomProposer, _mean_drive, rollout_branches

ROOT = Path(__file__).resolve().parent.parent
TEACHER = ROOT / "docs/doom/brains/squeeze2/seed0/cma_interface.json"
RUNS = ROOT / "runs/dagger"
# 24/8 is the teacher: the best third of the proposals. 24/24 averages *all* of
# them, which is the brain's own mean answer with no search in it -- the control
# that says whether the look-ahead taught anything a denoiser would not have.
ARMS = {"search": (24, 8), "self": (24, 24)}
LEVEL, STEPS = "doom4", 600
HORIZON, NOISE = 12, 0.3


def student(level: str = LEVEL) -> WormBrain:
    """The brain that drives: deterministic, which is the form that goes on a robot."""
    brain = WormBrain.from_file(TEACHER, maps=level, task="doom", sensors="ideal")
    brain.trainable, brain.noise = None, 0.0
    return brain


def label(world: object, brain: object, candidates: int, keep: int) -> Drive:
    """What the search says the student should have done here.

    The rollouts need a stochastic brain to have anything to choose between, so
    the noise goes on for the look-ahead and off again for the driving. That
    only works because `WormBrain.noise` reaches the running simulator (23b).
    """
    if isinstance(brain, WormBrain):
        brain.noise = NOISE
    scores, firsts = rollout_branches(world, brain, candidates, HORIZON)  # type: ignore[arg-type]
    if isinstance(brain, WormBrain):
        brain.noise = 0.0
    order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
    return _mean_drive([firsts[i] for i in order[:keep]])


def distance(a: Drive, b: Drive) -> float:
    """The units the imitation error is measured in."""
    return abs(a.forward - b.forward) + abs(a.turn - b.turn)


Row = tuple[dict[str, float], Drive, Drive]


def walk(
    seed: int,
    every: int,
    candidates: int,
    keep: int,
    twice: bool,
    proposer: object | None = None,
) -> list[Row]:
    """Drive with the student; label each visited state once, or twice for the gate.

    `proposer` is what the rollouts draw from. It defaults to the student, and
    the gate passes a random one to measure the band a quantity with *no* state
    dependence produces at this sample size -- without it a ratio of 0.9 looks
    like a finding, and the sweep of stage 23b shows it is not.
    """
    brain = student()
    world = build_world(seed, LEVEL, "doom")
    brain.reset()
    obs = world.observe()
    out: list[Row] = []
    for tick in range(STEPS):
        channels = obs.as_channels()
        if tick % every == 0:
            source = proposer if proposer is not None else brain
            first = label(world, source, candidates, keep)  # each restores the snapshot
            second = label(world, source, candidates, keep) if twice else first
            out.append((dict(channels), first, second))
        obs = world.drive(drive_of(brain.act(channels), float(brain.fire)))
        if world.finished:
            break
    return out


def arm(seeds: range, every: int, candidates: int, keep: int, proposer: object | None) -> float:
    """One row: label each of the student's states twice and return within / between."""
    withins, betweens = [], []
    for seed in seeds:
        rows = walk(seed, every, candidates, keep, twice=True, proposer=proposer)
        firsts = [r[1] for r in rows]
        n = len(firsts)
        withins.append(st.fmean(distance(a, b) for _c, a, b in rows))
        pairs = [(firsts[i], firsts[j]) for i in range(n) for j in range(i + 1, n)]
        betweens.append(st.fmean(distance(a, b) for a, b in pairs))
    within, between = st.fmean(withins), st.fmean(betweens)
    ratio = within / between
    print(
        f"    {candidates:2d} candidates, best {keep:2d} averaged:  within {within:.3f}"
        f"  between {between:.3f}  ratio {ratio:.2f}",
        flush=True,
    )
    return ratio


def gate(seeds: range, every: int) -> bool:
    """Is the label a function of the state? Answer before paying for anything.

    The control is not decoration. Stage 23b's sweep had a *random* proposer --
    a quantity with no state dependence whatsoever -- score a ratio of 0.93 at
    this sample size, so a worm scoring 0.9 means nothing on its own. The worm
    has to sit below the band the control produces, not merely below 1.
    """
    print(f"{LEVEL}: the search labelling the student's own states, each one answered twice\n")
    widths = ((12, 1), (24, 8), (24, 24))
    print("  the worm proposing")
    worm_ratios = [arm(seeds, every, c, k, None) for c, k in widths]
    print("\n  the control: proposals from nothing")
    control = [arm(seeds, every, c, k, RandomProposer(seed=1)) for c, k in widths]

    floor = min(control)
    passes = [w < floor for w in worm_ratios]
    print(f"\n  the control never went below {floor:.2f}, so that is the bar\n")
    if any(passes):
        which = [f"{c}/{k}" for (c, k), ok in zip(widths, passes, strict=True) if ok]
        print(f"the label carries state-dependent structure at {', '.join(which)}: collect")
        return True
    print("the label is noise at the student's states too -- do not collect")
    return False


def collect(seeds: range, arm: str) -> None:
    """Record the labels, one demonstration per seed, in trajectory order."""
    candidates, keep = ARMS[arm]
    demos = []
    for seed in seeds:
        rows = walk(seed, 1, candidates, keep, twice=False)
        demo = Demonstration()
        for channels, chosen, _ in rows:
            demo.add(channels, chosen)
        demos.append(demo)
        print(f"  {arm}, seed {seed}: {len(demo)} labelled ticks", flush=True)
    out = RUNS / f"demos_{arm}.json"
    save_demos(demos, out)
    print(f"saved {sum(len(d) for d in demos)} ticks to {out.relative_to(ROOT)}")


def train(generations: int, arm: str, population: int = 16) -> None:
    """Move the weights towards the labels; the same optimiser stage 23 used."""
    from wormlab.connectome import load_cook2019
    from wormlab.connectome.mappings import interface_synapses

    demos = load_demos(RUNS / f"demos_{arm}.json")
    brain = student()
    brain.trainable = interface_synapses(load_cook2019(), "interface")
    start = np.asarray(brain.get_weights(), dtype=float)
    flat = [a for d in demos for a in d.actions]
    mean = _mean_drive(flat)
    constant = st.fmean(distance(a, mean) for a in flat)
    before = imitation_error(brain, demos)
    print(f"  untrained, this brain already scores {before:.4f}")
    print(f"  a constant at the labels' mean scores {constant:.4f} -- the bar to beat")

    def fitness(genome: np.ndarray) -> float:
        brain.set_weights(genome)
        return imitation_fitness(brain, demos)

    result = cma_es(
        fitness,
        start,
        CMAConfig(generations=generations, population=population, sigma=0.2, seed=0),
        on_generation=lambda g, _w, _f: print(
            f"  gen {g.generation:3d}  error {-g.best:.4f}  sigma {g.sigma:.3f}", flush=True
        ),
    )
    brain.set_weights(result.best_weights)
    brain.meta.update({"candidate": "worm", "distilled": {"from": f"dagger/{arm}"}})
    RUNS.mkdir(parents=True, exist_ok=True)
    brain.save(RUNS / f"student_{arm}.json")
    error = -result.best_fitness
    verdict = "beats the constant" if error < constant else "WORSE than the constant"
    print(
        f"  trained {before:.4f} -> {error:.4f}, constant {constant:.4f} -- {verdict}"
        f"   ({100 * (before - error) / before:.0f} % of the way it moved)"
    )


def benchmark(seeds: range) -> None:
    """The only number that matters: the student alone, no search at runtime."""
    arms = [("before, untaught", TEACHER)]
    arms += [(f"taught by {a}", RUNS / f"student_{a}.json") for a in ARMS]
    for name, path in arms:
        if not Path(path).exists():
            continue
        total = kills = 0.0
        for seed in seeds:
            brain = WormBrain.from_file(path, maps=LEVEL, task="doom", sensors="ideal")
            world = build_world(seed, LEVEL, "doom")
            tracker = RewardTracker(RewardConfig())
            run_brain_episode(world, brain, STEPS, tracker)
            total += sum(tracker.breakdown.values())
            kills += world.kills
        n = len(seeds)
        print(f"  {name:18s} reward {total / n:6.2f}   kills {kills / n:.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument(
        "--first-seed", type=int, default=3000, help="4000+ are maps nothing trained on"
    )
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--arm", choices=[*ARMS, "both"], default="both")
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--population", type=int, default=16)
    args = ap.parse_args()
    seeds = range(args.first_seed, args.first_seed + args.seeds)
    if args.gate:
        gate(seeds, args.every)
    arms = list(ARMS) if args.arm == "both" else [args.arm]
    if args.collect:
        for a in arms:
            collect(seeds, a)
    if args.train:
        for a in arms:
            print(f"\n{a}:")
            train(args.generations, a, args.population)
    if args.benchmark:
        benchmark(seeds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
