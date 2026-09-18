"""Stage 23b: is the teacher a function of the state, before anything is built on it.

    uv run python scripts/teacher_probe.py

Stage 23 tried to distil the search and found there was nothing to distil: the
same state gave a different answer every time, so the demonstrations carried no
structure and a fitted constant beat the trained student. The fix under test is
consensus — average the best k branches instead of keeping the single luckiest
one — and the question is whether that is enough to make the teacher learnable.

The measurement asks it directly, and costs minutes rather than a training run.
At every decision the same state is searched *twice*, from the same neural
snapshot with fresh draws:

    within   how far the two answers to the *same* state are apart -- the noise
    between  how far answers to *different* states are apart -- the signal

Their ratio is the whole result. At 1.0 the teacher's action says nothing about
the state and no student can imitate it. Below it, the part that is a function
of the state is real and worth teaching.

The random proposer is the control the method file now demands: if it scores the
same ratio, whatever structure the consensus found belongs to the rollout and
not to the connectome.
"""

from __future__ import annotations

import argparse
import copy
import statistics as st
import sys
from pathlib import Path

from wormlab.body import Drive, drive_of
from wormlab.candidates import WormBrain
from wormlab.environments.worlds import build_world
from wormlab.learning.reward import RewardConfig
from wormlab.learning.search import RandomProposer, _mean_drive, _score_branch

ROOT = Path(__file__).resolve().parent.parent
TEACHER = ROOT / "docs/doom/brains/squeeze2/seed0/cma_interface.json"
LEVEL, STEPS, EVERY = "doom4", 300, 3
HORIZON, NOISE = 12, 0.3


def distance(a: Drive, b: Drive) -> float:
    """The units the imitation error is measured in: forward and turn."""
    return abs(a.forward - b.forward) + abs(a.turn - b.turn)


def decide(world: object, brain: object, k: int, candidates: int) -> Drive:
    """One decision: draw candidates, roll them forward, average the best k."""
    scored = [
        _score_branch(copy.deepcopy(world), brain, HORIZON, RewardConfig(), None)  # type: ignore[arg-type]
        for _ in range(candidates)
    ]
    order = sorted(range(len(scored)), key=lambda i: scored[i][0], reverse=True)
    return _mean_drive([scored[i][1] for i in order[:k]])


def probe(
    label: str, make_brain: object, k: int, seed: int, candidates: int = 12, every: int = EVERY
) -> tuple[float, float]:
    """Walk one episode; at each decision answer the same state twice."""
    brain = make_brain()  # type: ignore[operator]
    world = build_world(seed, LEVEL, "doom")
    brain.reset()
    world.observe()
    held, firsts, seconds = Drive(), [], []

    for tick in range(STEPS):
        if tick % every == 0:
            state = brain.sim.snapshot()
            a = decide(world, brain, k, candidates)
            brain.sim.restore(state)
            b = decide(world, brain, k, candidates)  # same state, fresh draws
            brain.sim.restore(state)
            firsts.append(a)
            seconds.append(b)
            held = a
        world.drive(held)
        if world.finished:
            break

    within = st.fmean(distance(a, b) for a, b in zip(firsts, seconds, strict=True))
    pairs = [(firsts[i], firsts[j]) for i in range(len(firsts)) for j in range(i + 1, len(firsts))]
    between = st.fmean(distance(a, b) for a, b in pairs)
    # between^2 = within^2 + signal^2 if the two are independent, so what is left
    # over after the noise is accounted for is the part that answers the world
    signal = (between**2 - within**2) ** 0.5 if between > within else 0.0
    print(
        f"  {label:26s} k={k:<3d} within {within:.3f}  between {between:.3f}"
        f"  ratio {within / between:.2f}  signal {signal:.3f}   ({len(firsts)} decisions)",
        flush=True,
    )
    return within, between


def silent(seed: int, driver: str = "deterministic", draws: int = 48, every: int = 8) -> None:
    """Does the noise blur the worm's answer, or erase what it was answering?

    Both are measured at the *same* states, on the deterministic trajectory, so
    neither can be explained by the noisy brain wandering somewhere duller:

        deterministic   the spread of the answer across states, noise off
        noisy mean      the same, averaged over many noisy draws from the same
                        neural snapshot -- the state-dependent part that survives

    If the second is much smaller than the first, the noise is not a veil over a
    policy: it moves the neurons somewhere the world no longer reaches them.
    """
    brain = worm()
    world = build_world(seed, LEVEL, "doom")
    brain.reset()
    obs = world.observe()
    det: list[Drive] = []
    drive_noise = 0.0 if driver == "the deterministic brain" else NOISE
    means: list[tuple[Drive, Drive]] = []

    for tick in range(STEPS):
        channels = obs.as_channels()
        if tick % every == 0:
            state = brain.sim.snapshot()
            pair = []
            for _ in range(2):  # twice, so the estimate carries its own error bar
                brain.noise = NOISE
                sample = []
                for _ in range(draws):
                    brain.sim.restore(state)
                    sample.append(drive_of(brain.act(channels), float(brain.fire)))
                pair.append(_mean_drive(sample))
            brain.noise = drive_noise
            brain.sim.restore(state)
            means.append((pair[0], pair[1]))
        brain.noise = drive_noise
        if driver == "the search":
            state = brain.sim.snapshot()
            intent = decide(world, brain, 1, 12)
            brain.sim.restore(state)
        else:
            intent = drive_of(brain.act(channels), float(getattr(brain, "fire", 0.0)))
        det.append(intent)
        obs = world.drive(intent)
        if world.finished:
            break

    keep = det[::every][: len(means)]
    print(f"  driven by {driver}")
    for label, acts in (("what it did", keep), ("noisy mean", [m[0] for m in means])):
        pairs = [(acts[i], acts[j]) for i in range(len(acts)) for j in range(i + 1, len(acts))]
        print(f"  {label:26s} spread across states {st.fmean(distance(*p) for p in pairs):.3f}")
    within = st.fmean(distance(a, b) for a, b in means)
    print(f"  {'':26s} same state, twice    {within:.3f}   ({len(means)} states, range is 4.0)")


def worm() -> WormBrain:
    brain = WormBrain.from_file(TEACHER, maps=LEVEL, task="doom", sensors="ideal")
    brain.trainable, brain.noise = None, NOISE
    return brain


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=3000)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 6, 12])
    ap.add_argument("--drivers-only", action="store_true", help="skip the consensus table")
    args = ap.parse_args()
    if args.drivers_only:
        print(f"{LEVEL}, seed {args.seed}: how much the answer varies where each driver goes")
        for driver in ("the deterministic brain", "the noisy brain", "the search"):
            silent(args.seed, driver)
        return 0
    print(f"{LEVEL}, seed {args.seed}: the same state searched twice, 12 candidates\n")
    print("the worm's proposals")
    for k in args.k:
        probe("consensus of the best", worm, k, args.seed)
    print("\nthe control: proposals from nothing")
    for k in (args.k[0], args.k[-1]):
        probe("consensus of the best", lambda: RandomProposer(seed=1), k, args.seed)
    print("\naveraging the noise down: no selection at all, just the mean of many draws")
    for n in (24, 48):
        probe("the mean draw", worm, n, args.seed, candidates=n, every=8)
    probe("the mean draw, from nothing", lambda: RandomProposer(seed=1), 48, args.seed, 48, 8)
    print("\nhow much the answer varies across the states each driver visits")
    for driver in ("the deterministic brain", "the noisy brain", "the search"):
        silent(args.seed, driver)
    return 0


if __name__ == "__main__":
    sys.exit(main())
