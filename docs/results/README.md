# Results

One folder per phase. Every report names the rows it was computed from; rows
are the JSON + CSV the benchmark writes. Trained brains live in `docs/brains/`.

| Phase | Report | Rows |
|---|---|---|
| A1 platform | `a1/stage11_comparison_2026-09-15.md` (real vs random vs shuffled, decision gate §17.1) | `a1/compare_2026-09-15/` |
| | `a1/stage12_13_eval_2026-09-15.md`, `a1/stage14_danger_eval_2026-09-15.md`, `a1/stage15_apartment_eval_2026-09-15.md`, `a1/stage17_sensors_eval_2026-09-15.md` | |
| | `a1/stage20_needs_eval_2026-09-15.md` (needs arbitration, phase A1 closed) | `a1/benchmark_2026-09-15/` |
| A2 candidates | `a2/a2_bakeoff_2026-09-15.md` (final three-seed table on the vacuum preset, decision for A3) | `a2/benchmark_a2_2026-09-15/` |
| A3 machine | `a3/stage22_car_2026-09-15.md` (car preset: transfer, retraining, layer v2) | `a3/benchmark_car_2026-09-15/`, `a3/benchmark_car_v2_2026-09-16/` |
| | stage 22.1 drive log demo | `a3/stage22_drive_2026-09-15/` |

Brains: `docs/brains/a1/` (stages 4-15), `docs/brains/a2/` (bake-off, seeds 0-2),
`docs/brains/car/` (retrained on the car preset, seeds 0-2). The dense (free)
topology brain is not committed (11 MB); its curve and meta are.
