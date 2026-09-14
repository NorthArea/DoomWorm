# DoomWorm

Experimental project: the **C. elegans connectome** (302 neurons) used as a trainable
controller, moving step by step from a 3-neuron network to a 2D agent and finally to a
small car that explores an apartment on its own needs. The worm is the reactive and
motivational layer; map building and planning are an engineered layer outside the brain.
Doom is an optional stress-test track.

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
uv run doomworm play --brain runs/small_evolved.json --seed 1003 --plot
```

## Layout

```text
src/doomworm/
  brain/           Neuron, Synapse, Network, Simulator
  connectome/      loader, internal graph model, name mappings
  environments/    simple_2d, maze, doom
  adapters/        sensory (obs -> stimulation), motor (activity -> actions)
  learning/        fitness, evolution, plasticity
  experiments/     baseline, random_network, real_connectome
  visualization/   brain activity, metrics, debug screen
tests/             pytest suite (one test module per layer)
docs/              assumptions, stage checklist
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
