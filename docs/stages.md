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
| 8 | Motor mapping | | | todo |
| 9 | C. elegans controls 2D agent | | | todo |
| 10 | Train weights | | | todo |
| 11 | Real vs random vs shuffled topology | | | todo |
| 12 | Random maps | | | todo |
| 13 | Target instead of food ("come to X") | | | todo |
| 14 | Danger zone | | | todo |
| 15 | Apartment-like maps (rooms, doors) | | | todo |
| 16 | Battery + dock | | | todo |
| 17 | Sensor noise, delay, hardware-like config | | | todo |
| 18 | External occupancy-grid map | | | todo |
| 19 | Planner -> virtual gradient, "come to X" on unseen map | | | todo |
| 20 | Needs arbitration: battery > call > curiosity | | | todo |
| 21 | Physical platform | | | todo |

Decision gate after stage 11 (Plan §17.1): record the real/random/shuffled/free
comparison in `docs/` and decide which brain goes forward.

## Track B: Doom (optional, after the decision gate)

| # | Stage | Status |
|---|-------|--------|
| B1 | Mini-Doom (2D) | todo |
| B2 | FIRE action | todo |
| B3 | Mini-Doom combat | todo |
| B4 | ViZDoom integration | todo |
| B5-B11 | Doom levels 1-7 | todo |
| B12 | Visual input | todo |
