# DoomWorm

Experimental project: the **C. elegans connectome** (302 neurons) as a trainable
controller, built up from a 3-neuron network to a simulated home vacuum robot, and
compared on one benchmark against a Roomba-style controller, networks trained from
scratch, PPO, Neural Circuit Policies and hybrids. Two tracks share the platform:
A = home vacuum (first), B = Doom player. Map building and planning are an engineered
layer outside the brain.

```text
Environment -> Sensory Adapter -> Brain Simulator -> Motor Adapter -> Environment
```

The research question: does the biological topology of C. elegans provide a useful
inductive bias for learning behaviour in a completely foreign environment, compared with
a same-size network trained from scratch?
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
uv run doomworm play --brain runs/small_evolved.json --seed 1003 --plot
```

## Fresh clone

```bash
uv sync --all-groups && uv run pre-commit install
make check                                                   # lint, mypy, 200+ tests
make demo-20                                                 # stage-20 benchmark rows (worm + scripted driver)
```

Trained brains live in `docs/results/brains/`; `runs/` is git-ignored scratch.

## Layout

```text
src/doomworm/
  brain/           Neuron, Synapse, Network, Simulator
  connectome/      loader, internal graph model, name mappings
  environments/    simple_2d, maze, doom
  adapters/        sensory (obs -> stimulation), motor (activity -> actions)
  learning/        fitness, evolution, reward, benchmark, bake-off harness, PPO
  mapping/         occupancy grid, BFS / coverage planner (engineered layer)
  brains/          Brain interface and candidates: worm, rnn, ncp, roomba, planner layer, needs
  hardware/        stage 22: robot link (sim | machine), units, wire protocol, teleop, log compare
  experiments/     baseline, random_network, real_connectome
  visualization/   brain activity, metrics, debug screen
tests/             pytest suite (one test module per layer)
docs/              assumptions, stage checklist, hardware contract (docs/hardware.md), results
firmware/          stage 22: ESP32 sketch speaking the robot protocol (not compiled yet)
data/connectome/   Cook 2019 connectome CSVs (see its README for source and citation)
scripts/           cross-cutting utilities
runs/              experiment outputs (git-ignored)
```

## Rules of the road

1. Stages are implemented strictly in the order of Plan §44, one new complexity at a time.
2. A stage is done only when it works, is tested, and has a runnable demo.
3. No CNN/Transformer between the environment and the worm. Adapters are simple transforms.
4. Connectome topology is fixed; only synaptic weights are trained.
5. Deterministic seeds everywhere; every experiment must be replayable.

## License

MIT
