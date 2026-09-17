# DoomWorm — project rules for agents

Source of truth: `Plan.md`. Read it before any implementation work.

The project is a test bed: the natural network (the *C. elegans* connectome)
against networks trained from scratch, playing Doom, under identical
conditions. Levels, budgets and seeds are *conditions*, not goals; the
deliverable is `docs/findings.md`, one row per question with the worm's number,
the best self-trained number, the difference and a verdict. Every measurement
that changes a row updates that file.

The platform (world, engine, sensing, brain contract, body, benchmark) is
finished before any new brain is written; then every candidate — the worm, its
topology controls, a net from scratch, an NCP, PPO, hybrids, the hand-written
floor — is compared only through the benchmark.

## Workflow
- Implement stages in the order of `docs/stages.md`. Never start stage N+1
  until stage N works, is covered by tests and has a runnable demo.
- Test first. Each layer has its test module in `tests/`.
- Minimal implementation per stage; prefer the simpler option when in doubt.
- Record every non-obvious decision in `docs/assumptions.md`, with the
  measurement behind it; tick the stage in `docs/stages.md`.

## Hard constraints
- No CNN/transformer/RL framework between the world and the brain. Adapters are
  simple transforms.
- Brain, connectome, environments, adapters and learning are independent
  layers. No game logic in the brain.
- Connectome topology is FIXED; only synaptic weights are trainable.
- Neuron model is leaky integrate-and-fire, activity graded in [0, 1]. No
  Hodgkin-Huxley, no NEURON, no body model.
- The network holds only the 302 connectome neurons: no artificial
  interneurons. Internal state is injected as currents into real neurons.
- Several brain ticks per environment step; the motor adapter averages over
  that window.
- A brain returns an intent (`Drive`), never actuators. The body turns it into
  wheels or engine buttons.
- Reward is the single source of numbers; fitness is the sum of reward over an
  episode. Evaluate on several seeded maps the brain has never seen.
- CPU + NumPy only. No GPU/CUDA/Rust/distributed until a result demands it.
- The simulator is never more convenient than the game. When the two disagree,
  the engine is right.
- No framebuffer until the vision stage, and then only through a deliberately
  small encoder.
- Deterministic seeds; experiments save seed, brain, weights and world, and can
  be replayed.

## Where a design choice is settled
By measurement. Where a signal lands, which neuron pair carries it, what gain
it gets — these are chosen by running the alternatives and reporting the
numbers, never by the nicest story. What the network *does* with a signal is
never hand-written.

## Tooling
- `uv` for everything: `uv sync --all-groups`, `uv run pytest`,
  `uv run ruff check . --fix`, `uv run ruff format .`, `uv run mypy`.
  `Makefile` wraps them: `make check`, `make demo-*`, `make watch-doom`.
- src layout: code in `src/doomworm/`, tests in `tests/`, cross-cutting scripts
  in `scripts/`, stage demos next to the stage code or under `experiments/`.
- Experiment outputs go to `runs/` (git-ignored); published rows to `docs/`.
- Strict mypy and ruff must pass before a stage is done.
- Never edit the source while a training lane is running.
