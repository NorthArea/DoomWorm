# Stage checklist

A stage is done only when it works, is covered by tests, and has an observable demo (Plan §45).

| # | Stage | Tests | Demo | Status |
|---|-------|-------|------|--------|
| 0 | 3-neuron network (Neuron, Synapse, Network, Simulator) | `tests/test_brain.py` | `python -m doomworm.experiments.three_neurons` | done |
| 1 | 2D agent: 3 distance sensors, 2 motors, 1 obstacle | `tests/test_simple_2d.py`, `tests/test_adapters.py`, `tests/test_obstacle_agent.py` | `python -m doomworm.experiments.obstacle_agent --plot` | done |
| 2 | Food + hunger | `tests/test_simple_2d.py`, `tests/test_food_agent.py` | `python -m doomworm.experiments.food_agent --plot` | done |
| 3 | Reward | `tests/test_reward.py`, `tests/test_episode.py`, `tests/test_reward_demo.py` | `python -m doomworm.experiments.reward_demo` | done |
| 4 | Evolution on the small network | `tests/test_evolution.py`, `tests/test_serialization.py`, `tests/test_evolve_small.py` | `doomworm train` then `doomworm play --brain runs/small_evolved.json --seed 1003` | done |
| 5 | Load C. elegans connectome into internal format | `tests/test_connectome_loader.py`, graded neuron tests in `tests/test_brain.py` | `python -m doomworm.experiments.connectome_stats` | done |
| 6 | Stimulate connectome, observe propagation | `tests/test_stimulation.py`, `tests/test_visualization.py` | `doomworm stimulate ASHL --plot`, `python -m doomworm.experiments.debug_screen_demo` | done |
| 7 | Sensory mapping | `tests/test_sensory_mapping.py` | `make demo-7` | done |
| 8 | Motor mapping | `tests/test_motor_mapping.py` | `make demo-8` | done |
| 9 | C. elegans controls 2D agent | `tests/test_worm_agent.py` | `make demo-9 SEED=1001` | done (untrained: drives, slows on touch, no reversal, no steering) |
| 10 | Train weights | `tests/test_evolve_worm.py`, `tests/test_simulator_vectorised.py` | `make train-worm` then `make play BRAIN=runs/worm_evolved.json SEED=1003` | done |
| 11 | Real vs random vs shuffled topology | `tests/test_variants.py` | `make compare` (see `docs/comparison.md`) | done: Real ~= Random; worm stays primary, controls kept |
| 12 | Random maps | `tests/test_random_maps.py` | `make train-worm-random`, `make demo-12 SEED=2002` | done (weak: ~1 food per unseen map, no collisions; see `docs/results/stage12_13_eval_2026-09-15.md`) |
| 13 | Target instead of food ("come to X") | `tests/test_target.py` | `make demo-13 SEED=2004` | done (partial transfer: 3 targets on 6 unseen maps without retraining) |
| 14 | Danger zone | `tests/test_danger.py` | `make train-worm-danger`, `make demo-14 SEED=2003` | done (avoidance learned: 0 damage, 0 deaths; target attraction weak; see `docs/results/stage14_danger_eval_2026-09-15.md`) |
| 15 | Apartment-like maps (rooms, doors) | `tests/test_apartment.py` | `make demo-15 SEED=3002` | done (weak: doors passed on 2/6 unseen maps by the stage-12 brain; apartment training made it worse; see `docs/results/stage15_apartment_eval_2026-09-15.md`) |
| 16 | A1: dirt map + coverage, battery, dock | `tests/test_vacuum.py` | `make demo-16 SEED=3002` | done (stage-12 brain runs unchanged: coverage 4%, 11 dockings; it treats the dock as food) |
| 17 | A1: vacuum sensor suite with noise and delay | `tests/test_sensors.py` | `doomworm play --brain runs/worm_evolved_random.json --maps random --sensors noisy --seed 2002` | done (ideal 0 collisions -> vacuum 171, noisy 104; see `docs/results/stage17_sensors_eval_2026-09-15.md`) |
| 18 | A1: Brain interface, Gymnasium env, benchmark + leaderboard | `tests/test_brain_interface.py` | `make benchmark BRAIN=runs/worm_evolved_random.json` | done |
| 19 | A1: occupancy grid, path + coverage planner -> virtual gradient | `tests/test_mapping.py` | `doomworm benchmark --brain runs/worm_evolved_random.json --planner coverage` | done (planner doubles coverage: worm 7% -> 14%, driver 12% -> 24%; nobody docks yet) |
| 20 | A1: needs arbitration (battery > call > clean) | `tests/test_needs.py` | `make demo-20` | done: on unseen apartments with vacuum sensors the driver recharges and survives 16/18 episodes (reward v2 +57, coverage 28%), the worm 16/18 (reward v2 +40, coverage 20%, after the stage-21.9 layer fixes); calls reach the next room on the vacuum preset; see `docs/results/stage20_needs_eval_2026-09-15.md`. **Phase A1 closed 2026-09-15** (Plan §47 milestone). |
| 21 | A2: bake-off of all candidate brains on the benchmark | | `make benchmark-a2` | done, sub-stages below |
| 21.1 | A2: Roomba-style classical controller (zero learning, the floor) | `tests/test_roomba.py` | `make demo-21` | done: reward v2 +29.5 (v1 −0.1), coverage 19%, survives 13/18 (`docs/results/benchmark_2026-09-15/roomba.csv`) |
| 21.2 | A2: generic training harness (any weight-trainable Brain under the same layer, same evolution, parallel fitness); retrain worm + controls on the vacuum task | `tests/test_bakeoff.py` | `make evolve-a2 CANDIDATE=worm` then `doomworm benchmark --brain runs/a2/worm.json --planner needs` | done: all five worm rows trained under reward v2 and benchmarked (`docs/results/a2_bakeoff_2026-09-15.md`) |
| 21.3 | A2: small recurrent net from scratch, same evolution | `tests/test_rnn.py` | `make evolve-a2 CANDIDATE=rnn` then `doomworm benchmark --brain runs/a2/rnn.json --planner needs` | done: 34.5 ± 6.4 over 3 seeds after the stage-21.10 forward bias (ties the worm from scratch) |
| 21.4 | A2: PPO via Gymnasium (optional `rl` group) | `tests/test_rl.py` (skipped without the group) | `uv sync --group rl` then `doomworm ppo` and `doomworm benchmark --brain runs/a2/ppo.json --planner needs` | done: best learned brain, reward 49.7, coverage 0.31, survives 9/18 |
| 21.5 | A2: Neural Circuit Policies (optional `rl` group) | `tests/test_ncp.py` (skipped without the group) | `make evolve-a2 CANDIDATE=ncp` then `doomworm benchmark --brain runs/a2/ncp.json --planner needs` | done: 28.8 ± 2.8 over 3 seeds after the stage-21.10 forward bias (at the Roomba floor) |
| 21.6 | A2: hybrids | | | worm under the planner (hybrid 1 of Plan §20.4) is every `+needs` row; other hybrids not built |
| 21.8 | A2: dock autopilot in the planner layer (the layer drives the trip to the dock, the brain only cleans) | `tests/test_needs.py` | `make benchmark-a2` | done: survival 0.93-0.98 for every brain; PPO 61.8 ± 4.4, survives 50/54 |
| 21.9 | A2: bumper reflex in the planner layer; map footprint fixed to the body radius | `tests/test_needs.py` | `make benchmark-a2` | done: PPO 61.9 ± 0.2 with 8 collisions; curriculum worm 44.0 ± 7.6; real 32.1 > random 15.9; final table in `docs/results/a2_bakeoff_2026-09-15.md` |
| 21.10 | A2: forward bias for rnn / ncp (untrained nets stood still or spun) | `tests/test_rnn.py`, `tests/test_ncp.py` | `make evolve-a2 CANDIDATE=rnn` | done: both learn from generation 0; rnn 34.5 ± 6.4, ncp 28.8 ± 2.8 |
| 21.7 | A2: leaderboard with >= 3 seeds per candidate, decision for A3 | | `make benchmark-a2` | done: 3 seeds for the six main rows; final (21.9-21.10): PPO 61.9 ± 0.2 > worm-curriculum 44.0 ± 7.6 > shuffled 40.4 > rnn 34.5 > real 32.1 > Roomba 29.5 > ncp 28.8 > dense 26.2 > random 15.9; decision for A3 = engineered layer + PPO, worm stays the control (`docs/results/a2_bakeoff_2026-09-15.md`). **Phase A2 closed 2026-09-15** (second milestone, Plan §47). |
| 22 | A3: physical vacuum (Plan §20.5) | | | in progress, sub-stages below; no hardware in hand yet |
| 22.1 | A3: robot link (simulator and machine behind one loop), units contract, wire protocol, teleop + sensor recording, log replay against the simulator | `tests/test_hardware.py` | `make demo-22` | done (software side; `docs/hardware.md`, `docs/results/stage22_drive_2026-09-15/`) |
| 22.1b | A3: platform = ACEBOTT QD001 + QD003 car (Plan §20.5.1); sensor preset `car` (servo sweep, proximity bumper, binary wall IR, command odometry, no gyro, camera marker as the dock beacon); per-preset calibration; host-side dead reckoning; A2 brains benchmarked on it without retraining | `tests/test_sensors.py`, `tests/test_hardware.py` | `make benchmark-car` | done: worm-curriculum 41.3 > worm 38.8 > driver 34.9 > PPO 31.9 > rnn 26.7 > Roomba 24.6 (`docs/results/stage22_car_2026-09-15.md`) |
| 22.1c | A3: candidates retrained on the `car` preset, three seeds, three-seed table with the transfer rows | | `make evolve-car CANDIDATE=...`, `make benchmark-car` | done: PPO-car 43.0 ± 4.0 ≈ transferred curriculum worm 41.2 ± 0.2 (14 vs 31 collisions) > worm 34.3 > rnn-car 30.1 ≈ worm-car 30.1 > PPO transferred 27.3 > rnn 27.2 > Roomba 24.6; car brain = transferred curriculum worm, PPO-car second (`docs/results/stage22_car_2026-09-15.md`) |
| 22.1d | A3: everything for day one prepared on the simulator: `doomworm selftest` (protocol, sensors at rest, latency, wheel/odometry directions), `doomworm calibrate` (metres per unit, wheel base, tick from two tape measurements -> `--calibration`), room files (`rooms/*.json`, metres; `--room` on every link command, `build_world("room:<file>")`, the room travels inside the log), `doomworm plot-log`; day-one checklist in `docs/hardware.md` | `tests/test_hardware.py` | `make selftest-sim` | done |
| 22.1e | A3: layer v2 for the camera marker (line of sight in the model; homing on any sighting; dock re-anchored from every sighting; spin search near the estimated dock) | `tests/test_needs.py`, `tests/test_sensors.py` | `make benchmark-car` | done: survival driver 0.61 -> 0.89, PPO-car 0.81 -> 0.89, worm 0.80 -> 0.94; rewards up 1-12; decision unchanged (`docs/results/stage22_car_2026-09-15.md` §3) |
| 22.2 | A3: firmware on the car (`firmware/esp32_car/`, written, not compiled); self-test, calibration, first real teleop log; sim-vs-real report; `CAR` preset corrected from it | | `doomworm selftest --link tcp`, `doomworm calibrate --link tcp`, `doomworm drive --link tcp --teleop --record ...`, `doomworm compare-log` | todo: needs the car |
| 22.3 | A3: obstacle avoidance with the best car-preset brain under the layer | | | todo |
| 22.4 | A3: marker "dock" through the K210 (return to the start spot) | | | todo |
| 22.5 | A3: coverage, map, call on the car (acceptance of Plan §20.5.1); the vacuum acceptance of §20.5 stays the project target | | | todo |

Known platform limitation carried into A2: the occupancy grid has no scan matching, so odometry drift (~1 unit per 300 ticks on the vacuum preset) shifts doors on the map; this costs 2-3 of 18 episodes per brain.

Candidate brains for stage 21 (Plan §20.4): Roomba-style classical controller,
worm (connectome + evolution), worm controls (random / shuffled / dense),
small recurrent net from scratch, PPO via Gymnasium, Neural Circuit Policies,
hybrids.

Decision gate after stage 11 (Plan §17.1): recorded in `docs/comparison.md` on
2026-09-15. Decision: the connectome stays the primary brain; random, shuffled
and dense variants are trained as controls in every later run.

## Track B: Doom player (after phase A1)

| # | Stage | Status |
|---|-------|--------|
| B1 | ViZDoom platform: Gymnasium env, structured observations, scenarios (Plan §21-33) | todo |
| B2 | Candidate brains on the Doom benchmark | todo |
| B3 | Visual input (Plan §34) | todo |
