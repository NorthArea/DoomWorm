# One repository, two tracks — rules for agents

This repository holds two projects that share one nervous system and one
platform. Read `Plan.md` and the track's own `docs/<track>/stages.md` before
implementation work, and `docs/knowledge/` before an experiment.

```text
src/wormlab/      the platform both tracks share. Changes rarely, and a change
                  here must keep both tracks green.
src/doomworm/     the Doom player. Changes fast.
src/broomworm/    the home robot. Changes fast.
docs/knowledge/   what either track learned that the other can use.
docs/doom/        the Doom track's findings, stages, assumptions, results.
docs/broom/       the robot track's.
```

`make tracks` prints this map. `make help` lists every target; the robot
track's are prefixed `broom-`.

## The question, in both tracks

The natural network (the *C. elegans* connectome) against networks trained from
scratch, under identical conditions. The task differs — one plays Doom, one
drives a floor robot — the nervous system does not. Each track's deliverable is
its own `findings.md`: one row per question, with the worm's number, the best
self-trained number, the difference and a verdict. Every measurement that
changes a row updates that file.

## Working in two tracks at once

- **Touch one track's folder per change.** Anything in `src/wormlab/` is shared:
  changing it means running both tracks' tests, and saying so in the commit.
- **A track never edits the sibling's `docs/<track>/` or `src/<track>/`.**
- **Promote a finding to `docs/knowledge/` when it is true without naming the
  task**, and only with the numbers and the run it came from. "Restricting the
  genome to the interface synapses beats tuning all 5905 at the same budget" is
  knowledge; "the worm cannot reach the exit on doom2" is a track result.
- **Read the sibling's knowledge before an experiment.** Several entries there
  exist because one track spent hours on something the other would repeat.
- Commit messages name the track they touch.

## Workflow

- Implement stages in the order of the track's `docs/<track>/stages.md`. Never
  start stage N+1 until stage N works, is covered by tests and has a demo.
- Test first. Each layer has its test module in `tests/`.
- Minimal implementation per stage; prefer the simpler option when in doubt.
- Record every non-obvious decision in `docs/<track>/assumptions.md`, with the
  measurement behind it; tick the stage in `docs/<track>/stages.md`.

## Hard constraints (both tracks)

- No CNN/transformer/RL framework between the world and the brain. Adapters are
  simple transforms.
- Brain, connectome, environments, adapters and learning are independent
  layers. No task logic in the brain.
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
- The simulator is never more convenient than the real thing — the game for one
  track, the machine for the other. When they disagree, reality is right.
- No framebuffer until a vision stage, and then only through a deliberately
  small encoder.
- Deterministic seeds; experiments save seed, brain, weights and world, and can
  be replayed. A published row must replay from what is committed.

## Where a track-specific thing belongs

A sensor preset describes a *machine*, so it lives in the track that owns the
machine and registers itself with `register_preset` (see
`src/broomworm/presets.py`). A level, an engine, a hand-written floor: the
track. The connectome, the simulator, the body, the search, the benchmark: the
platform.

## Where a design choice is settled

By measurement. Where a signal lands, which neuron pair carries it, what gain
it gets — these are chosen by running the alternatives and reporting the
numbers, never by the nicest story. What the network *does* with a signal is
never hand-written.

## Tooling

- `uv` for everything: `uv sync --all-groups`, `uv run pytest`,
  `uv run ruff check . --fix`, `uv run ruff format .`, `uv run mypy`.
  `Makefile` wraps them: `make setup`, `make check`, `make tracks`.
- The CLI is `wormlab` (`doomworm` still works).
- Experiment outputs go to `runs/` (git-ignored); published rows to
  `docs/<track>/`.
- Strict mypy and ruff must pass before a stage is done.
- Never edit the source while a training lane is running, and never snapshot
  `docs/<track>/brains/` until the lane reports it is done — both cost us a day
  once (`docs/knowledge/method.md`).
