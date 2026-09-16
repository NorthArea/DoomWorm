# DoomWorm — project rules for agents

Source of truth: `Plan.md`. Read it before any implementation work.

The project is a test bed (Plan §0): the natural network (C. elegans connectome) against
networks trained from scratch, under identical conditions. Environments, sensor presets,
budgets and hardware are *conditions* (axes of Plan §20.7), not goals; the deliverable is
`docs/findings.md`, one row per axis with the worm's number, the best self-trained number,
the profit and a verdict. Every measurement that changes a row updates that file.

Two tracks on one platform: A = home robot (vacuum -> kit car), B = Doom player (after A1).
In both, the platform (world, sensor emulation, battery/dock, Brain interface, Gymnasium
env, benchmark) is finished before any new brain is written; then every candidate brain
(Roomba-style controller, worm, controls, net from scratch, PPO, NCP, hybrids) is compared
only through the benchmark (Plan §3.3, §20.4). Map and planner are an engineered layer
outside the brain (Plan §3.2). Torch-based candidates live in an optional uv group.

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
- The simulator models the future hardware (range sensors, metres, noise); never make it
  more convenient than the real world (Plan §2.4). No hardware before stage 17.
- Doom (track B) only after the decision gate in Plan §17.1, and only if asked.
- Deterministic seeds; experiments save seed/brain/weights/environment and can be replayed.

## Tooling
- `uv` for everything: `uv sync --all-groups`, `uv run pytest`, `uv run ruff check . --fix`,
  `uv run ruff format .`, `uv run mypy`. `Makefile` wraps them: `make check`, `make demo-N`,
  `make train`, `make play`, `make stimulate`. Add a `demo-N` target for every new stage.
- src layout: code in `src/doomworm/`, tests in `tests/`, cross-cutting scripts in `scripts/`,
  stage demos live next to the stage code as `demo_*.py` or under `experiments/`.
- Experiment outputs go to `runs/` (git-ignored).
- Strict mypy and ruff must pass before a stage is considered done.
