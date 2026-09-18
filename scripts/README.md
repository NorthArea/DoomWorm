Cross-cutting utilities (run with `uv run scripts/<name>.py`).
Stage-specific demos live next to the stage code, not here.

| Script | What it does |
|---|---|
| `fetch_connectome.py` | download the connectome dataset |
| `archive_runs.py` | turn a lane's raw benchmark rows into a table that outlives `runs/`, into `docs/doom/results/` |
| `doom_report.py` | build the tables in `docs/results/b/` from the benchmark rows |
| `doom_b2_resume.py` | the bake-off lane: train every candidate on `doom4` and `doom6`, three seeds, benchmark each on all six levels and in the engine. Idempotent — re-run it to fill gaps after an interrupted run |
| `search_probe.py` | stage 21: the worm alone, the worm made stochastic, and a search over its proposals, on the same maps |
| `doom_b2d_lane.py` | the same lane for the worm family after the trigger and prey fixes (stage 12) |
