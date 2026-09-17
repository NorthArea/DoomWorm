# Results

The tables here are what `docs/findings.md` is computed from. Rows are the JSON
and CSV the benchmark writes; trained brains live in `docs/brains/`.

| Report | Rows |
|---|---|
| `b/b2_doom_2026-09-16.md` — every candidate on the Doom benchmark, three seeds, simulator and engine | `runs/benchmark_doom/` (regenerate with `scripts/doom_report.py`) |

Brains: `docs/brains/a1/` are the curriculum brains from the food task, the
ancestor every "curriculum worm" row starts from; `docs/brains/doom/` are the
Doom-trained ones. Results for the vacuum robot and the kit car live in the git
history up to `1bf247a`.
