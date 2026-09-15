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
| 15 | Apartment-like maps (rooms, doors) | | | todo |
| 16 | Battery + dock | | | todo |
| 17 | Sensor noise, delay, hardware-like config | | | todo |
| 18 | External occupancy-grid map | | | todo |
| 19 | Planner -> virtual gradient, "come to X" on unseen map | | | todo |
| 20 | Needs arbitration: battery > call > curiosity | | | todo |
| 21 | Physical platform | | | todo |

Decision gate after stage 11 (Plan §17.1): recorded in `docs/comparison.md` on
2026-09-15. Decision: the connectome stays the primary brain; random, shuffled
and dense variants are trained as controls in every later run.

## Track B: Doom (optional, after the decision gate)

| # | Stage | Status |
|---|-------|--------|
| B1 | Mini-Doom (2D) | todo |
| B2 | FIRE action | todo |
| B3 | Mini-Doom combat | todo |
| B4 | ViZDoom integration | todo |
| B5-B11 | Doom levels 1-7 | todo |
| B12 | Visual input | todo |
