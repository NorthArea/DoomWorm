# DoomWorm

A test bed for one question:

> **Does the nervous system of *C. elegans* — 302 neurons, wired the way the
> animal is actually wired — play Doom better than a network of the same size
> trained from scratch?**

The connectome is the candidate. A small recurrent net, a Neural Circuit
Policy, PPO and a hand-written reflex player are the yardsticks. Everything is
compared through one benchmark, on seeded levels nobody tuned against, and
every number that changes a verdict lands in `docs/findings.md`.

## What is here

```text
the world            a 2D simulator with walls, monsters, a gun and an exit,
                     and the same seeded layouts run by the real Doom engine
                     through ViZDoom (no framebuffer: structured observations)
the brain contract   channels in, Drive(forward, turn, strafe, fire) out
the body             a vehicle turns that intent into actuators or engine buttons
the candidates       the connectome, its topology controls (random, shuffled,
                     dense), a small recurrent net, an NCP, PPO, a hybrid, and
                     the zero-learning `doomguy` floor
the training         evolution on the synaptic weights; the topology is fixed
the food task        the simple attractant world the worm is pre-trained on
                     before it sees a level (the curriculum of stage 5)
```

The worm senses Doom the way a worm senses anything: walls as touch, the exit
as a smell, monsters as pain — and, since stage 6, as prey. It has no map, no
memory of where it has been, and no image. See `docs/findings.md` for what that
buys and what it costs.

## Run it

```bash
uv sync --all-groups          # core + dev; add --group doom for the engine
make check                    # ruff, mypy, pytest
make demo-b1 LEVEL=doom4      # the hand-written floor on a simulator level
make demo-b4 LEVEL=doom4      # the same level inside the Doom engine
make watch-doom LEVEL=doom4   # watch a trained worm play, in a real window
make watch-doomguy            # watch the floor, which actually aims
make demo-b11 STOCK=stock_defend   # a scenario shipped with ViZDoom
```

Training and benchmarking:

```bash
uv run doomworm evolve --candidate worm --maps doom4 --out runs/worm.json
uv run doomworm benchmark --brain runs/worm.json --maps doom6
uv run doomworm ppo --maps doom4          # needs the rl group
uv run doomworm play --brain runs/worm.json --maps vizdoom4 --watch
```

## Where the numbers are

| File | What it holds |
|---|---|
| `docs/findings.md` | the map: one row per question, with the worm's number, the best self-trained number and a verdict |
| `docs/stages.md` | what is built, what is next, and how to reproduce each row |
| `docs/assumptions.md` | every non-obvious decision, dated, with the measurement behind it |
| `docs/results/b/` | the benchmark tables the rows are computed from |
| `docs/brains/` | trained brains, committed so a row can be replayed |

## The rules this project keeps

- The connectome's **topology is fixed**; only synaptic weights are trained.
- The neuron model is leaky integrate-and-fire, activity graded in [0, 1]. No
  Hodgkin-Huxley, no body model.
- **No CNN, transformer or RL framework between the world and the brain.**
  Adapters are simple transforms; PPO is a candidate, not the platform.
- The brain never sees game logic, and the world is never made more convenient
  than the game.
- Deterministic seeds. Every experiment saves its seed, brain, weights and
  world, and can be replayed.
- A measurement beats an argument. When a design choice is open, it is settled
  by running it, and the numbers go in `docs/assumptions.md`.
