"""Stage 27: can this wiring turn towards something off to the side?

    uv run python scripts/face_probe.py

Stage 26 priced the hand-written floor's reflexes and found the worm already has
three of the four that pay: the exit gradient is chemotaxis, contact reversal is
the animal's own, and the trigger has worked since stage 12. The missing one is
`face` -- come round onto a target that is not already ahead -- and it is worth
37 points, the most expensive thing this brain has never shown.

This asks whether the wiring can express it at all, with no world in the way.
The channel is held at a constant value on one side and the intent is read back:

    sign      does a signal on the left produce a turn to the left
    size      how much turn, in the units the body actually uses
    hold      does it persist while the signal does, or adapt away -- the last
              third of the window against the first
    stop      does it fall back when the signal moves to the front, or spin on

A `face` reflex needs all four. Chemotaxis needs none of them: a klinokinesis
animal turns *randomly* and simply turns less when things improve, which is a
different circuit and would show here as sign near zero with plenty of size.

Controls, because a difference between two sides could be anything: the same
weights on a shuffled connectome, a random topology of the same size, and the
untrained wiring. If the real one does not separate from those, there is nothing
here that training failed to find -- there is nothing here.
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

from wormlab.body import drive_of
from wormlab.candidates import CandidateSpec, WormBrain, build_candidate

ROOT = Path(__file__).resolve().parent.parent
B2D = ROOT / "docs/doom/brains/doom_b2d/doom6"
SQUEEZE = ROOT / "docs/doom/brains/squeeze2/seed0/cma_interface.json"
TICKS = 60
LOUDNESS = (0.05, 0.1, 0.17, 0.3, 0.6, 1.0)
QUIET = {"health": 1.0, "ammo": 1.0}


def hold(brain: WormBrain, channel: str, ticks: int = TICKS, value: float = 1.0) -> list[float]:
    """Hold one channel steady and read the turn the brain asks for, each tick."""
    brain.reset()
    channels = dict(QUIET)
    channels[channel] = value
    out = []
    for _ in range(ticks):
        intent = drive_of(brain.act(channels), float(getattr(brain, "fire", 0.0)))
        out.append(intent.turn)
    return out


def face(brain: WormBrain, pair: tuple[str, str]) -> dict[str, float]:
    """The four numbers a `face` reflex needs, for one pair of side channels."""
    left, right = (hold(brain, c) for c in pair)
    third = len(left) // 3
    early = st.fmean(left[:third]) - st.fmean(right[:third])
    late = st.fmean(left[-third:]) - st.fmean(right[-third:])
    front = hold(brain, pair[0].replace("left", "front"))
    return {
        "sign": st.fmean(left) - st.fmean(right),  # > 0: a left signal turns left
        "size": max(abs(st.fmean(left)), abs(st.fmean(right))),
        "hold": late / early if abs(early) > 1e-9 else 0.0,
        "stop": abs(st.fmean(front)),  # small: it settles when the target is ahead
    }


def arms(seed: int) -> list[tuple[str, WormBrain]]:
    """The trained worm and the controls it has to separate from."""
    kw = {"maps": "doom6", "task": "doom", "sensors": "ideal"}
    out: list[tuple[str, WormBrain]] = []
    for name in ("worm", "worm_shuffled", "worm_random"):
        path = B2D / f"seed{seed}" / f"{name}.json"
        if path.exists():
            out.append((name, WormBrain.from_file(path, **kw)))  # type: ignore[arg-type]
    out.append(("untrained", build_candidate(CandidateSpec("worm"))))  # type: ignore[arg-type]
    if seed == 0 and SQUEEZE.exists():
        out.append(("squeeze2", WormBrain.from_file(SQUEEZE, **kw)))  # type: ignore[arg-type]
    return out


def gain(seeds: list[int], channels: str) -> None:
    """The turn against how loud the signal is -- the game never sends it at 1.0.

    The sector channel is 1/distance, so a monster at the gun's range arrives at
    about 0.17 and one across the room at 0.05. A circuit that only answers a
    signal pinned at full is a circuit that never fires in play.
    """
    pair = (f"{channels}_left", f"{channels}_right")
    print(f"\n  how loud the signal has to be ({pair[0]} against {pair[1]})\n")
    print("  brain            " + "".join(f"{v:>9.2f}" for v in LOUDNESS))
    rows: dict[str, list[list[float]]] = {}
    for seed in seeds:
        for name, brain in arms(seed):
            got = []
            for value in LOUDNESS:
                left = st.fmean(hold(brain, pair[0], value=value))
                right = st.fmean(hold(brain, pair[1], value=value))
                got.append(left - right)
            rows.setdefault(name, []).append(got)
    for name, got in rows.items():
        mean = [st.fmean(g[i] for g in got) for i in range(len(LOUDNESS))]
        print(f"  {name:14s} " + "".join(f"{v:+9.3f}" for v in mean))
    print("\n  in play the sector reads ~0.17 at the gun's range, ~0.05 across the room")


USEFUL = (0.05, 0.17, 0.30)  # what the sector actually reads in play


def face_score(brain: WormBrain, pair: tuple[str, str], front: str) -> float:
    """One number for the whole reflex: turn towards the side, and settle when centred.

    Averaged over the loudness the game really sends, because a circuit tuned to
    a signal pinned at 1.0 is a circuit that never fires at six units away.
    """
    total = 0.0
    for value in USEFUL:
        left = st.fmean(hold(brain, pair[0], value=value))
        right = st.fmean(hold(brain, pair[1], value=value))
        centre = st.fmean(hold(brain, front, value=value))
        total += 0.5 * (left - right) - abs(centre)
    return total / len(USEFUL)


def train_face(kind: str, generations: int, channels: str) -> None:
    """Can *any* weights on this topology do it? The question stage 26 left open."""
    import numpy as np

    from wormlab.connectome import load_cook2019
    from wormlab.connectome.mappings import interface_synapses
    from wormlab.learning.cmaes import CMAConfig, cma_es

    pair = (f"{channels}_left", f"{channels}_right")
    front = f"{channels}_front"
    brain = build_candidate(CandidateSpec(kind, variant_seed=1))
    brain.trainable = interface_synapses(load_cook2019(), "interface")
    start = np.asarray(brain.get_weights(), dtype=float)
    print(f"  {kind:14s} untrained {face_score(brain, pair, front):+.3f}", end="", flush=True)

    def fitness(genome: np.ndarray) -> float:
        brain.set_weights(genome)
        return face_score(brain, pair, front)

    result = cma_es(
        fitness, start, CMAConfig(generations=generations, population=24, sigma=0.3, seed=0)
    )
    brain.set_weights(result.best_weights)
    got = face(brain, pair)
    print(
        f"   ->  {result.best_fitness:+.3f}"
        f"   (sign {got['sign']:+.3f}, stop {got['stop']:.3f}, hold {got['hold']:.2f})",
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--channels", default="prey", choices=["prey", "target", "danger"])
    ap.add_argument("--gain", action="store_true", help="sweep how loud the signal is")
    ap.add_argument("--train", action="store_true", help="ask whether any weights can do it")
    ap.add_argument("--generations", type=int, default=120)
    args = ap.parse_args()
    if args.train:
        print(f"training for `face` alone on {args.channels}, no world, nothing else scored\n")
        for kind in ("worm", "worm_shuffled", "worm_random"):
            train_face(kind, args.generations, args.channels)
        return 0
    if args.gain:
        gain(args.seeds, args.channels)
        return 0
    pair = (f"{args.channels}_left", f"{args.channels}_right")
    print(f"a constant {pair[0]} against {pair[1]}, {TICKS} ticks, no world\n")
    print("  brain            sign      size     hold     stop   per seed")
    rows: dict[str, list[dict[str, float]]] = {}
    for seed in args.seeds:
        for name, brain in arms(seed):
            rows.setdefault(name, []).append(face(brain, pair))
    for name, got in rows.items():
        mean = {k: st.fmean(g[k] for g in got) for k in got[0]}
        signs = [g["sign"] for g in got]
        # per seed as well as the mean: a mean sign hides a seed that disagrees,
        # and three seeds agreeing is 1 in 8 by chance, which is not much
        agree = "all agree" if len({x > 0 for x in signs}) == 1 else "SPLIT"
        print(
            f"  {name:14s} {mean['sign']:+8.3f}  {mean['size']:8.3f}"
            f" {mean['hold']:8.2f} {mean['stop']:8.3f}   "
            + " ".join(f"{x:+.3f}" for x in signs)
            + f"  {agree}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
