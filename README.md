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
the world            a 2D simulator with walls, monsters, a gun and an exit;
                     the same seeded layouts run by the real Doom engine through
                     ViZDoom; the scenarios ViZDoom ships; and the classic
                     game's own maps, e1m1..e4m9, from the Freedoom data
                     (no framebuffer anywhere: structured observations only)
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
make setup                    # venv, the connectome, the Doom data — everything
make check                    # ruff, mypy, pytest
```

Then look at it:

```bash
make demo-b1 LEVEL=doom4           # the hand-written floor on a simulator level
make demo-b4 LEVEL=doom4           # the same level inside the Doom engine
make demo-b11 STOCK=stock_defend   # a scenario shipped with ViZDoom
make demo-classic CLASSIC=e1m1     # the classic game's own first map
```

Watch it play, in a real Doom window at the game's own speed:

```bash
make watch-doomguy            # the floor, the one player that actually aims
make watch-doom LEVEL=doom4   # a trained worm
make watch-classic CLASSIC=e1m1
```

Train and measure:

```bash
make evolve-doom CANDIDATE=worm LEVEL=doom4   # one brain
make benchmark-doom LEVEL=doom6               # every brain of a seed, on a level
make lane-bakeoff                             # the full three-seed bake-off (hours, resumable)
make lane-memory                              # stage 16: with and without the memory layer
make report                                   # rebuild the tables in docs/results/
```

`make help` lists every target.

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
