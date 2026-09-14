# DoomWorm — project rules for agents

Source of truth: `Plan.md`. Read it before any implementation work.

## Workflow
- Implement stages strictly in the order of Plan §44. Never start stage N+1 until stage N
  works, is covered by tests, and has a runnable demo (Plan §45).
- Test first. Each layer has a test module in `tests/` (see Plan §43 for required test names).
- Minimal implementation per stage; prefer the simpler option when in doubt (Plan §51).
- Record every non-obvious decision in `docs/assumptions.md`; tick stages in `docs/stages.md`.

## Hard constraints
- No CNN/Transformer/RL framework between environment and brain. Adapters are simple transforms.
- Brain, connectome, environments, adapters, learning are independent layers. No game logic in the brain.
- Connectome topology is FIXED; only synaptic weights are trainable.
- Neuron model v1 is Leaky Integrate-and-Fire. No Hodgkin-Huxley/NEURON/body model.
  Binary activity on stages 0-4; graded activity in [0, 1] from stage 5 (Plan §2.3).
- From stage 5 the network holds only the 302 connectome neurons: no artificial
  interneurons; internal states are injected as currents into biological neurons (Plan §11).
- Several brain ticks per environment step (`brain_steps_per_env_step`, Plan §3.1);
  the motor adapter averages over that window.
- Reward is the single source of numbers; fitness = sum of reward over an episode (Plan §9-10).
  Evaluate on several seeded maps with food respawn, never on one fixed layout.
- CPU + NumPy only. No GPU/CUDA/Rust/distributed until the hypothesis is demonstrated.
- No ViZDoom until the 2D agent works; no framebuffer until structured-input Doom works.
- Deterministic seeds; experiments save seed/brain/weights/environment and can be replayed.

## Tooling
- `uv` for everything: `uv sync --all-groups`, `uv run pytest`, `uv run ruff check . --fix`,
  `uv run ruff format .`, `uv run mypy`.
- src layout: code in `src/doomworm/`, tests in `tests/`, cross-cutting scripts in `scripts/`,
  stage demos live next to the stage code as `demo_*.py` or under `experiments/`.
- Experiment outputs go to `runs/` (git-ignored).
- Strict mypy and ruff must pass before a stage is considered done.
