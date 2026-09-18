# Results: the rows every finding is computed from

`runs/` is the scratch desk and gets cleared. These tables are what survives
it, written by `uv run python scripts/archive_runs.py` straight from the
benchmark rows a lane produced. Each folder holds the aggregated table and the
lane's own log; the brains that produced them are in `docs/doom/brains/`.

Every number is the mean over a row's episodes, then over seeds; the spread is
across seeds.

| Lane | What it measured | Feeds |
|---|---|---|
| `b2_bakeoff` | every candidate on `doom1`..`doom6` and, unchanged, in the engine — three seeds, the first full table | findings 1-3 (the nets' rows still stand) |
| `b2d_shooting` | the worm family retrained after the two anatomy fixes: a trigger it can reach and a target it can turn to | findings 1-2, re-measured |
| `b2d_asg_superseded` | the same lane with the first, wrong prey pathway (ASG). Kept because the choice between ASG and CEPD was made on these numbers | nothing; it is the losing arm of a comparison |
| `memory` | a place memory as a layer, with the shuffled control, trained under the condition | findings 5 (the hypothesis that failed) |
| `squeeze_genome` | how much of the connectome the search may move: all 5905, the 1819 interface, the 506 sensory outputs | findings 6 |
| `squeeze_search` | sep-CMA-ES against fixed-sigma mutation, and the adapter gains as genes | findings 7, the ceiling |
| `hybrid_c7` | a frozen brain as a reflex module under a small net, against that net alone | findings 5 (hybrids) |
| `budget_pilot` | 12 training maps and 60 generations instead of 3 and 25 | the budget entry in `docs/knowledge/search.md` |

`b/b2_doom_2026-09-16.md` is the hand-written report of the first bake-off,
with the protocol and the transfer matrices in full.
