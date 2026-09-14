# DoomWorm

Experimental project: the **C. elegans connectome** (302 neurons) used as a trainable
controller, moving step by step from a 3-neuron network to a 2D agent and finally to Doom.

```text
Environment -> Sensory Adapter -> Brain Simulator -> Motor Adapter -> Environment
```

The research question: does the biological topology of C. elegans provide a useful
inductive bias for learning behaviour in a completely foreign environment?
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

```bash
uv run pytest                 # tests
uv run pytest --cov           # tests + coverage
uv run ruff check . --fix     # lint
uv run ruff format .          # format
uv run mypy                   # type-check
uv run doomworm --help        # CLI
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
