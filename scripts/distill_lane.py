"""Stage 23: record the teacher, teach the brain, then measure the brain alone.

    uv run python scripts/distill_lane.py --collect      # episodes of the search
    uv run python scripts/distill_lane.py --train        # move the weights to match
    uv run python scripts/distill_lane.py --benchmark    # the student, with no search

The point is the third step. A rollout over the connectome's proposals beats
the connectome (stage 21); if what it found can be taught back, then the
architecture could always express that behaviour and only the search was
missing. If it cannot, the limit is the wiring. Nothing else in this project
separates those two.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from wormlab.candidates import WormBrain
from wormlab.environments.worlds import build_world
from wormlab.episode import run_brain_episode
from wormlab.learning.cmaes import CMAConfig, cma_es
from wormlab.learning.distill import Demonstration, imitation_fitness, load_demos, save_demos
from wormlab.learning.reward import RewardConfig, RewardTracker
from wormlab.learning.search import SearchConfig, searched_episode

ROOT = Path(__file__).resolve().parent.parent
TEACHER = ROOT / "docs/doom/brains/squeeze2/seed0/cma_interface.json"
DEMOS = ROOT / "runs/distill/demos.json"
STUDENT = ROOT / "runs/distill/student.json"
LEVEL, STEPS = "doom4", 600


def fresh(level: str = LEVEL) -> WormBrain:
    brain = WormBrain.from_file(TEACHER, maps=level, task="doom", sensors="ideal")
    brain.trainable = None  # set by the caller when a restricted genome is wanted
    return brain


def collect(seeds: range) -> None:
    cfg = SearchConfig(candidates=6, horizon=12, plan_every=1, noise=0.3)
    demos: list[Demonstration] = []
    for seed in seeds:
        brain = fresh()
        brain.noise, brain.episode = cfg.noise, seed
        world = build_world(seed, LEVEL, "doom")
        demo = Demonstration()

        def record(row: object, into: Demonstration = demo) -> None:
            into.add(row.channels, row.chosen_drive)  # type: ignore[attr-defined]

        reward, _rows = searched_episode(world, brain, STEPS, cfg, on_decision=record)
        demos.append(demo)
        print(f"  seed {seed}: reward {reward:6.2f}, {len(demo)} demonstrated ticks", flush=True)
    save_demos(demos, DEMOS)
    print(f"saved {sum(len(d) for d in demos)} ticks to {DEMOS.relative_to(ROOT)}")


def train(generations: int) -> None:
    from wormlab.connectome import load_cook2019
    from wormlab.connectome.mappings import interface_synapses

    demos = load_demos(DEMOS)
    brain = fresh()
    # the same cut stage 18 measured as the best value per evaluation
    brain.trainable = interface_synapses(load_cook2019(), "interface")
    start = np.asarray(brain.get_weights(), dtype=float)

    def fitness(genome: np.ndarray) -> float:
        brain.set_weights(genome)
        return imitation_fitness(brain, demos)

    print(f"teaching {brain.n_weights} weights on {sum(len(d) for d in demos)} ticks")
    result = cma_es(
        fitness,
        start,
        CMAConfig(generations=generations, population=16, sigma=0.2, seed=0),
        on_generation=lambda g, _w, _f: print(
            f"  gen {g.generation:3d}  error {-g.best:.4f}  sigma {g.sigma:.3f}", flush=True
        ),
    )
    brain.set_weights(result.best_weights)
    brain.meta.update({"candidate": "worm", "distilled": {"from": str(TEACHER.name)}})
    STUDENT.parent.mkdir(parents=True, exist_ok=True)
    brain.save(STUDENT)
    print(f"student error {-result.best_fitness:.4f} -> {STUDENT.relative_to(ROOT)}")


def benchmark(seeds: range) -> None:
    for label, path in (("teacher, alone", TEACHER), ("student, alone", STUDENT)):
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
        print(f"  {label:16s} reward {total / n:6.2f}   kills {kills / n:.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--generations", type=int, default=30)
    args = ap.parse_args()
    seeds = range(3000, 3000 + args.seeds)
    if args.collect:
        collect(seeds)
    if args.train:
        train(args.generations)
    if args.benchmark:
        benchmark(seeds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
