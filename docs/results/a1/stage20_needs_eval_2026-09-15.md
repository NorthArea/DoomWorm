# Stage 20: needs arbitration on the benchmark

Benchmark (apartment / clean / vacuum sensors, 6 unseen maps x 3 noise repeats,
800 steps, battery drains in 500 ticks). All six rows of
`benchmark_2026-09-15/leaderboard.md` were re-run on the final stage-20 code
(the sensor delay buffer changed, see below), so every row matches the code:

| brain | reward v2 | reward v1 | coverage | collisions | dockings | survived | ticks |
|---|---|---|---|---|---|---|---|
| driver_follower+needs | 55.58 ± 15.13 | 12.22 ± 9.28 | 0.28 | 10.8 | 3.0 | 16/18 | 767 |
| worm_stage12+needs | 19.64 ± 17.48 | -4.91 ± 13.60 | 0.16 | 93.6 | 2.7 | 15/18 | 751 |
| driver_follower+planner | 34.86 ± 8.21 | -5.63 ± 3.64 | 0.27 | 6.3 | 1.5 | 0/18 | 538 |
| worm_stage12+planner | 4.64 ± 15.61 | -20.09 ± 10.00 | 0.16 | 96.9 | 1.7 | 2/18 | 582 |

Reward v1 is the scale these rows were first published with; reward v2
(stage 21: a cleaned cell 0.5 instead of 0.1, docking bonus once per
discharge cycle) is what `leaderboard.md` now shows. The brains and their
behaviour are identical; only the score changed.

With arbitration both brains recharge and keep cleaning until the 800-step
cap in most episodes; without it (stage-19 rows) they discharge at ~540 ticks.
The worm still collides ten times more than the scripted driver: that is the
brain, not the platform, and is what the A2 bake-off is for.

## What had to change to get there

The first stage-20 run (same code as the handoff commit) survived 0/18
(driver) and 2/18 (worm) episodes on the vacuum preset, while the driver
survived 15/18 on the ideal preset. Three causes, fixed in order:

1. **Sensor delay returned zeros at start.** With `delay` 1 the first tick
   reported odometry (0, 0), and the arbiter stored the dock there. The
   suite now pre-fills its delay buffer with the reading at rest.
2. **Dead reckoning drifts about 1 unit per 300 ticks** (`odom_sigma` 0.05,
   gyro 0.002 rad/tick), so the stored dock pose is off by the time the
   robot returns. Fix: the planner layer follows the dock beacon
   (`dock_*`, exact bearing) once its strength is >= 0.5 (within 2 units),
   treats a rising battery as "on the dock" and re-anchors the dock estimate
   there. Every trip in which beacon homing engaged ended in a docking.
3. **A fixed battery threshold (0.4 = 200 ticks) is too late from the far
   room.** The arbiter now leaves when `battery <= low + cost_per_cell *
   path_cells_to_dock` (0.008 per 0.5-unit cell, measured on the driver).

Remaining failures (2/18 driver, 3/18 worm) are map drift: the occupancy
grid lives in the odometry frame, a door mapped early and re-mapped later
smears or shifts by ~1 unit against a 2.4-unit opening, and the planned path
runs into the jamb until the battery is gone. Fixing that needs scan
matching / loop closure (Plan §3.2 "classical SLAM"); left for A2/A3 if a
candidate needs it.

## Diagnostic: the same driver on the ideal preset

`broomworm benchmark --driver --planner needs --sensors ideal` (noiseless
suite, exact odometry, repeats identical; one row per map):

| map | ticks | coverage | collisions | dockings | survived |
|---|---|---|---|---|---|
| 3000 | 504 | 0.14 | 4 | 1 | 0 |
| 3001 | 800 | 0.21 | 17 | 4 | 1 |
| 3002 | 800 | 0.35 | 9 | 2 | 1 |
| 3003 | 800 | 0.21 | 7 | 4 | 1 |
| 3004 | 800 | 0.27 | 20 | 2 | 1 |
| 3005 | 800 | 0.35 | 6 | 4 | 1 |

Mean over 18 episodes: reward 10.03 ± 11.17, coverage 0.25, 2.8 dockings,
survival 0.83. Map 3000 fails on ideal but not on vacuum: with exact odometry
the trip-cost rule sends the robot home later on that layout and it runs out
of charge on the way (one seed, not investigated further).

Reproduce: `make demo-20` (vacuum rows), the ideal command above; the
per-trip breakdown used for the diagnosis is not a repository script.
