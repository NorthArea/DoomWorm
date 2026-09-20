Cross-cutting utilities (run with `uv run scripts/<name>.py`).
Stage-specific demos live next to the stage code, not here.

| Script | What it does |
|---|---|
| `fetch_connectome.py` | download the connectome dataset |
| `distill_lane.py` | stage 23: record the search as a teacher, train the brain towards it, measure the brain alone |
| `audit_brains.py` | which published brains anything still points at, and whether they all still load |
| `face_probe.py` | stage 27: can the wiring turn towards something off to the side — sign, gain, hold, stop |
| `ablation_lane.py` | stage 26: price each of the hand-written floor's reflexes, and what the benchmark can read |
| `reach_lane.py` | stage 25: the denominator every row was missing, and what a policy with no behaviour scores |
| `reach_probe.py` | stage 25: sweep the search's look-ahead — at what depth does each part of the reward appear |
| `dagger_lane.py` | stage 24: the student drives, the search labels — with a gate that refuses to collect until the labels are a function of the state |
| `teacher_probe.py` | stage 23b: ask the same state twice — is a teacher a function of the state, and where |
| `competence_map.py` | stage 22: play with the proposal search on and group the decisions by situation, into `docs/doom/results/competence/` |
| `archive_runs.py` | turn a lane's raw benchmark rows into a table that outlives `runs/`, into `docs/doom/results/` |
| `doom_report.py` | build the tables in `docs/results/b/` from the benchmark rows |
| `doom_b2_resume.py` | the bake-off lane: train every candidate on `doom4` and `doom6`, three seeds, benchmark each on all six levels and in the engine. Idempotent — re-run it to fill gaps after an interrupted run |
| `search_probe.py` | stage 21: the worm alone, the worm made stochastic, and a search over its proposals, on the same maps |
| `doom_b2d_lane.py` | the same lane for the worm family after the trigger and prey fixes (stage 12) |
