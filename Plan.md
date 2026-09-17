# DoomWorm — plan

## 0. The question

Does the natural network beat a network trained from scratch?

The candidate is the *C. elegans* connectome: 302 neurons, the wiring of the
animal, taken as given. The yardsticks are a small recurrent net, a Neural
Circuit Policy, PPO, the worm's own topology controls, and a hand-written
reflex player with no learning at all. The task is Doom.

The deliverable is `docs/findings.md`: one row per question, with the worm's
number, the best self-trained number, the difference and a verdict. Levels,
budgets and hardware are *conditions*, not goals. A row is measured when the
numbers and the benchmark rows behind them are both in the file.

## 1. What the brain is

```text
sensors  ->  currents into named neurons  ->  302 LIF neurons  ->  motor groups
                                                                       |
                                          Drive(forward, turn, strafe, fire)
```

- Neuron model: leaky integrate-and-fire, graded activity in [0, 1].
- Several brain ticks per environment step (`brain_steps_per_env_step`); the
  motor adapter averages over that window.
- The topology is fixed. Training moves synaptic weights, nothing else.
- The sensory and motor tables are declared artifices and live in
  `connectome/mappings.py`. Where a signal lands is a design choice, settled by
  measurement; what the network does with it is never hand-written.

## 2. What the brain is not allowed to be

- No CNN, transformer or RL framework between the world and the brain.
- No game logic inside the brain: no "if an enemy is ahead, shoot".
- No image. Observations are structured numbers (Plan §4), and that holds in
  the real engine too — the framebuffer is off.
- No Hodgkin-Huxley, no NEURON, no body model.
- CPU and NumPy. No GPU, no distributed training, until a result demands it.

## 3. The world

Two worlds behind one contract:

- **The simulator** — a 2D world with walls, round obstacles, monsters that
  hurt in range with line of sight, a hitscan gun on a finite magazine, and an
  exit that ends the episode. Six seeded levels, `doom1`..`doom6`, rising from
  an empty room to an apartment with two walking monsters.
- **The engine** — the same seeded layouts written as a PWAD and played by
  ViZDoom in PLAYER mode, plus the scenarios that ship with it
  (`stock_defend`, `stock_corridor`, `stock_home`): maps nobody here drew. The
  engine owns motion, collisions, monsters and damage; the world mirrors its
  state and senses it with the platform's own code.

The simulator must never be more convenient than the engine. When the two
disagree, the engine is right and the simulator is corrected.

## 4. What the brain sees

| channel | neurons | what it is to a worm |
|---|---|---|
| `sensor_left/front/right` | ALM, FLP, AVM | touch: something close on that side |
| `target_*` | AWA, AWC, ASE | the exit as a smell, 1/distance |
| `danger_*` | ASH | a visible monster as pain |
| `prey_*` | CEPD | the same monster as prey, by side, while the gun is loaded |
| `aim` | ADL, RIP | how centred the nearest visible monster is on the gun line |
| `ammo`, `hunger` | NSM, ASI | internal state |

Motor groups: forward (AVB, PVC), reversal (AVA, AVD), turn (SMDD, SMDV, RIV),
trigger (the pharyngeal pumping group M3, M4, MC — biting as firing).

## 5. The body

The brain returns an intent, never actuators. A body turns it into what exists:
a wheel pair, four mecanum wheels, or the engine's buttons. A body that cannot
strafe ignores the component; a brain with no lateral gait leaves it at zero,
which is a fact about the animal rather than a gap in the platform.

## 6. The candidates

```text
doomguy        hand-written reflexes, zero learning. The floor every trained
               brain must beat.
worm           the connectome + evolution on the weights.
worm controls  random, shuffled and dense topologies, the same size, the same
               training. They say whether the *wiring* matters.
rnn            a small recurrent net from scratch, the same evolution.
ncp            Neural Circuit Policies (Lechner et al.), gradient trained.
ppo            PPO on the structured channels, another class of algorithm.
hybrid         a frozen brain as a reflex module under a trainable net.
```

A candidate joins by implementing the brain contract and passing the benchmark
without changing the platform.

## 7. The benchmark

One protocol for everyone: the same levels, the same sensor preset, the same
step budget, 12 unseen maps per row, at least three seeds. Fitness is the sum
of reward over an episode. Reward pays +10 for the exit, +10 per kill, +2 per
hit, +0.1 per newly covered cell, −2 per tick of damage taken, −20 for dying,
and a small toll for grinding along a wall.

Budgets are matched per method — generations for evolution, environment steps
for PPO — and the cost in raw experience is reported next to the result, not
hidden.

## 8. The curriculum

The worm is allowed to grow up before it fights: a simple attractant world
(`--maps random --task food`) where it learns to move toward a smell without
hitting things. A brain trained there is then dropped into a level unchanged.
Pre-training is a condition to be measured, not an advantage to assume — the
nets get their own equivalent or none, and the row says which.

## 9. Stages

Done, in the order they were built (the old `A*` / `B*` labels survive in the
git history):

1. The 2D world, the connectome loader, the LIF simulator, the adapters.
2. Evolution on the weights; the topology controls and the decision gate.
3. The Doom layer: monsters, the gun, the exit, six levels, the `doomguy` floor.
4. The engine: PWADs from the same layouts, ViZDoom behind the world contract.
5. The three-seed bake-off of every candidate, simulator and engine.
6. A stock ViZDoom scenario — a map nobody here drew.
7. The two anatomy fixes: a trigger the connectome can reach (through RIP, the
   pharynx's only door) and a target it can turn toward (prey on CEPD).
8. The vehicle leaves the brain's contract.

Next, in order:

9. **Re-measure the shooting rows** with the reachable trigger and the prey
   channel: the worm family against its controls and the nets, three seeds, on
   `doom4` and `doom6`, then in the engine unchanged.
10. **The reorientation problem.** The worm hunts what is in front of it and
    cannot turn onto a target 60 degrees off the bow: its own tonic turning
    bias is larger than any sensory current. In the animal that manoeuvre is
    the pirouette — a reversal and an omega turn driven by the gradient getting
    worse — and our neurons have leak but no derivative. Left to evolution
    first; a mapping change only if the numbers say search cannot find it.
11. **Strafe.** The engine has it and the worm does not. Whether the nets are
    given a fourth output the animal cannot have is a question for the map, not
    a bug to fix.
12. **Vision** (§11): the framebuffer through a very small encoder into
    sensory channels. Only after the structured rows are closed, and the
    encoder must never become the network that solves the task.

## 10. How a stage is finished

- Tests first, one module per layer, named in `docs/stages.md`.
- The smallest implementation that answers the question; when in doubt, the
  simpler option.
- A runnable demo (`make demo-*`) and a row in `docs/stages.md`.
- Every non-obvious decision, with the measurement behind it, in
  `docs/assumptions.md`.
- `make check` green: ruff, strict mypy, the whole suite.
- Never start the next stage while the current one is red.

## 11. Vision, later

```text
framebuffer -> a very small visual encoder -> sensory channels -> the connectome
```

The encoder stays small on purpose. If it grows into a network that could solve
the level by itself, the experiment has answered a different question.

## 12. Operating rules

- Deterministic seeds; an experiment saves seed, brain, weights and world.
- Experiment outputs go to `runs/` (git-ignored); published rows to `docs/`.
- Never edit the source while a training lane is running — an edit that is
  broken for ninety seconds has already cost six benchmark runs.
- Every number in a document has a source. If it was not measured, it says so.
