"""Stage 22: build the competence map from searched episodes.

    uv run python scripts/competence_map.py [--level doom4] [--seeds 6]

Plays the best brain we have with the proposal search switched on, logging every
decision, and writes the table into docs/doom/results/competence/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from wormlab.candidates import WormBrain
from wormlab.environments.worlds import build_world
from wormlab.learning.competence import table
from wormlab.learning.search import SearchConfig, SearchRow, searched_episode

ROOT = Path(__file__).resolve().parent.parent
BRAIN = ROOT / "docs/doom/brains/squeeze2/seed0/cma_interface.json"
OUT = ROOT / "docs/doom/results/competence"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level", default="doom4")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--candidates", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=12)
    args = ap.parse_args()

    cfg = SearchConfig(candidates=args.candidates, horizon=args.horizon, plan_every=1, noise=0.3)
    rows: list[SearchRow] = []
    rewards = []
    for seed in range(3000, 3000 + args.seeds):
        brain = WormBrain.from_file(BRAIN, maps=args.level, task="doom", sensors="ideal")
        brain.noise, brain.episode = cfg.noise, seed
        world = build_world(seed, args.level, "doom")
        reward, log = searched_episode(world, brain, args.steps, cfg)
        rewards.append(reward)
        rows.extend(log)
        print(f"  seed {seed}: reward {reward:6.2f}, {len(log)} decisions", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    header = (
        f"# Competence map — `{args.level}`\n\n"
        f"{len(rows)} decisions over {args.seeds} unseen maps, {args.candidates} proposals each,\n"
        f"rolled {args.horizon} ticks forward, replanning every tick. Mean episode reward\n"
        f"{sum(rewards) / len(rewards):.2f}.\n\n"
        "**brain already right** is how often the connectome's own first answer was the one\n"
        "the rollout kept. **what search added** is the score it gained over that answer.\n"
        "**how far it moved** is the distance between the two intents.\n\n"
    )
    (OUT / f"{args.level}.md").write_text(header + table(rows))
    print(f"\n{header}{table(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
