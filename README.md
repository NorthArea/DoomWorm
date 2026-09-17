# DoomWorm

A test bed: the **C. elegans connectome** (302 neurons, fixed wiring, trainable
weights) against networks trained from scratch (small RNN, PPO, NCP) and a Roomba-style
controller, under identical conditions: same simulated home, same sensor emulation, same
engineered layer (map, planner, needs), same training budget, one benchmark. Environments,
sensor presets and hardware are conditions, not goals. The result is the map of where the
natural network wins, ties or loses: **[docs/findings.md](docs/findings.md)**.

```text
Environment -> Sensory Adapter -> Brain Simulator -> Motor Adapter -> Environment
```

The research question: in which conditions does the natural network give the most profit
over a network one can train oneself, and how much? So far (2026-09-16): it loses to PPO on
the task it was trained for, ties a same-size RNN, and wins on transfer without retraining
when the sensors change (−6 % vs PPO's −56 %). Details and every number: `docs/findings.md`.
Full roadmap and constraints: [Plan.md](Plan.md). Stage progress: [docs/stages.md](docs/stages.md).

## Requirements

- Python 3.12+ (project pinned to 3.14 in `.python-version`)
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync --all-groups
uv run pre-commit install
```

## Development

`make help` lists every shortcut: `make check` runs lint, mypy and tests; `make demo-N`
runs a stage demo; `make train`, `make play SEED=1003`, `make stimulate NEURON=ASHL`.
The underlying commands:

```bash
uv run pytest                 # tests
uv run pytest --cov           # tests + coverage
uv run ruff check . --fix     # lint
uv run ruff format .          # format
uv run mypy                   # type-check
```

## Stage demos

```bash
uv run python -m doomworm.experiments.three_neurons        # stage 0
uv run python -m doomworm.experiments.obstacle_agent --plot # stage 1
uv run python -m doomworm.experiments.food_agent --plot     # stage 2
uv run python -m doomworm.experiments.reward_demo           # stage 3
uv run doomworm train                                       # stage 4: evolve weights
uv run python -m doomworm.experiments.connectome_stats     # stage 5: real connectome
uv run doomworm stimulate ASHL --plot                      # stage 6: propagation
uv run python -m doomworm.experiments.debug_screen_demo    # stage 6: debug screen GIF
uv run python -m doomworm.experiments.sensory_mapping_demo # stage 7: sensory mapping
uv run python -m doomworm.experiments.motor_mapping_demo   # stage 8: motor mapping
uv run python -m doomworm.experiments.worm_agent --plot --gif # stage 9: untrained worm drives
uv run doomworm train --scenario worm                      # stage 10: evolve connectome weights
uv run doomworm play --brain runs/worm_evolved.json --seed 1003 --plot
uv run doomworm compare                                    # stage 11: topology comparison
uv run doomworm train --scenario worm --maps random          # stage 12: random maps
make demo-13                                                # stage 13: come to a target
make train-worm-danger && make demo-14                      # stage 14: avoid a danger zone
make benchmark BRAIN=runs/worm_evolved_random.json          # stage 18: benchmark + leaderboard
make benchmark-a2                                           # stage 21: A2 bake-off of every candidate
make demo-22                                                # stage 22.1: drive over the robot link, replay the log
make selftest-sim                                           # stage 22.2 rehearsal: self-test, calibration, room drive
make demo-b1 LEVEL=doom4                                    # track B: mini-Doom level 4 (enemy + gun) on the simulator
make demo-b4 LEVEL=doom4                                    # track B: the same level in the Doom engine (uv sync --group doom)
uv run doomworm play --brain runs/small_evolved.json --seed 1003 --plot
```

## Fresh clone

```bash
uv sync --all-groups && uv run pre-commit install
make check                                                   # lint, mypy, 200+ tests
make demo-20                                                 # stage-20 benchmark rows (worm + scripted driver)
```

Trained brains live in `docs/brains/a1/`; `runs/` is git-ignored scratch.

## Layout

Code follows the layers of Plan §3: environment -> sensors -> engineered layer -> brain.

```text
src/doomworm/
  brain/           neuron model: Neuron, Synapse, Network, Simulator (LIF, graded)
  connectome/      C. elegans loader, internal graph, name mappings, topology controls
  adapters/        sensory (channels -> currents), motor (activity -> wheels)
  environments/    simple_2d world, maze/apartment maps, sensor suite + presets
                   (ideal | vacuum | noisy | car), worlds.py factories, gym_env.py,
                   doom/ (track B): mini-Doom levels, PWAD writer, the Doom engine world
  layer/           the engineered layer outside the brain: occupancy grid, path and
                   coverage planning, needs arbitration, PlannerLayer (dock autopilot,
                   bumper reflex, marker search), GradientFollower
  candidates/      the brains compared on the benchmark: worm, rnn, ncp, roomba, doomguy,
                   Brain / Trainable protocols, registry + load_candidate (PPO via rl group)
  learning/        reward, evolution, benchmark + leaderboard, bake-off harness,
                   PPO (rl group), multi-seed summary
  hardware/        stage 22: robot link (sim | tcp), units, wire protocol, teleop,
                   drive log, sim-vs-real compare, room files, selftest, calibrate, plot
  experiments/     stage demos and training entry points (stages 0-14)
  visualization/   brain rasters, debug screen
  episode.py       the closed loop for any brain;  cli.py  the doomworm command
tests/             one module per layer (test_brain, test_sensors, test_needs, test_hardware, ...)
docs/
  stages.md        stage checklist (what is done, how it is demonstrated)
  assumptions.md   every non-obvious decision, dated
  hardware.md      the machine contract: sensors, units, protocol, day-one checklist
  results/a1|a2|a3 reports and benchmark rows per phase (README.md there is the index)
  brains/a1|a2|car published trained brains: A1 stages, A2 bake-off, retrained for the car
data/connectome/   Cook 2019 connectome CSVs;  data/rooms/  real rooms for the simulator
firmware/          ESP32 sketch speaking the robot protocol (not compiled yet)
scripts/           cross-cutting utilities;  runs/  experiment outputs (git-ignored)
```

## Rules of the road

1. Stages are implemented strictly in the order of Plan §44, one new complexity at a time.
2. A stage is done only when it works, is tested, and has a runnable demo.
3. No CNN/Transformer between the environment and the worm. Adapters are simple transforms.
4. Connectome topology is fixed; only synaptic weights are trained.
5. Deterministic seeds everywhere; every experiment must be replayable.

## License

MIT
