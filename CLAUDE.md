# BroomWorm — project rules for agents

Source of truth: `Plan.md`. Read it before any implementation work.

The project is BroomWorm, a home vacuum robot (Plan §0). Two hardware phases: A = the
ACEBOTT QD001 ESP32 car with the QD003 vision pack (Plan §6, stages 23.x), B = a real vacuum
(Plan §7, stages 24.x). The simulation platform, candidates, benchmark and hardware link
built as stages 0-22.1 are the shared foundation (Plan §1); the previous plan with the
section numbers that docs/assumptions.md and the reports cite is
docs/history/plan_v1_doomworm.md. It is also a test bed: the natural network (C. elegans
connectome) against networks trained from scratch under identical conditions; the
deliverable is docs/findings.md, one row per axis of Plan §8. No Doom in this repository
(that track is NorthArea/DoomWorm); removing the leftover mini-Doom mechanics from the
simulator is stage 23.0.

Every candidate brain (Roomba-style controller, worm, controls, net from scratch, PPO, NCP,
hybrids) is compared only through the benchmark (Plan §3.3, §9). Map and planner are an
engineered layer outside the brain (Plan §3.1). Torch-based candidates live in the optional
uv group `rl`.

## Workflow
- Implement stages strictly in the order of Plan §6 then §7. Never start stage N+1 until
  stage N works, is covered by tests, and has a runnable demo (Plan §10).
- Test first. Each layer has a test module in `tests/` (Plan §10).
- Minimal implementation per stage; prefer the simpler option when in doubt.
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
  more convenient than the real world (Plan §2); hardware numbers come from measurements.
- Never add Co-Authored-By / Claude-Session or any agent attribution to commits.
- Deterministic seeds; experiments save seed/brain/weights/environment and can be replayed.

## Tooling
- `uv` for everything: `uv sync --all-groups`, `uv run pytest`, `uv run ruff check . --fix`,
  `uv run ruff format .`, `uv run mypy`. `Makefile` wraps them: `make check`, `make demo-N`,
  `make train`, `make play`, `make stimulate`. Add a `demo-N` target for every new stage.
- src layout: code in `src/broomworm/`, tests in `tests/`, cross-cutting scripts in `scripts/`,
  stage demos live next to the stage code as `demo_*.py` or under `experiments/`.
- Experiment outputs go to `runs/` (git-ignored).
- Strict mypy and ruff must pass before a stage is considered done.
