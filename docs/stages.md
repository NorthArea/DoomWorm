# Stage checklist

A stage is done only when it works, is covered by tests, and has an observable demo (Plan §45).

| # | Stage | Tests | Demo | Status |
|---|-------|-------|------|--------|
| 0 | 3-neuron network (Neuron, Synapse, Network, Simulator) | `tests/test_brain.py` | `python -m doomworm.experiments.three_neurons` | done |
| 1 | 2D agent: 3 distance sensors, 2 motors, 1 obstacle | `tests/test_simple_2d.py`, `tests/test_adapters.py`, `tests/test_obstacle_agent.py` | `python -m doomworm.experiments.obstacle_agent --plot` | done |
| 2 | Food + hunger | `tests/test_simple_2d.py`, `tests/test_food_agent.py` | `python -m doomworm.experiments.food_agent --plot` | done |
| 3 | Reward | | | todo |
| 4 | Evolution on the small network | | | todo |
| 5 | Load C. elegans connectome into internal format | | | todo |
| 6 | Stimulate connectome, observe propagation | | | todo |
| 7 | Sensory mapping | | | todo |
| 8 | Motor mapping | | | todo |
| 9 | C. elegans controls 2D agent | | | todo |
| 10 | Train weights | | | todo |
| 11 | Real vs random vs shuffled topology | | | todo |
| 12 | Random maps | | | todo |
| 13 | Target instead of food | | | todo |
| 14 | Danger / enemy | | | todo |
| 15 | Mini-Doom (2D) | | | todo |
| 16 | FIRE action | | | todo |
| 17 | Mini-Doom combat | | | todo |
| 18 | ViZDoom integration | | | todo |
| 19 | Doom L1: empty room, EXIT | | | todo |
| 20 | Doom L2: walls, corridors | | | todo |
| 21 | Doom L3: stationary enemy, avoid | | | todo |
| 22 | Doom L4: FIRE at stationary enemy | | | todo |
| 23 | Doom L5: moving enemy | | | todo |
| 24 | Doom L6: several rooms | | | todo |
| 25 | Doom L7: standard scenario | | | todo |
| 26 | Visual input | | | todo |
