# WormLab

Two projects, one nervous system.

> **Does the nervous system of *C. elegans* — 302 neurons, wired the way the
> animal is actually wired — beat a network of the same size trained from
> scratch?**

The connectome is the candidate in both of them. What differs is the task.

| Track | The task | Code | Docs |
|---|---|---|---|
| **DoomWorm** | play Doom: monsters, a gun, an exit, and the classic game's own maps | `src/doomworm/` | `docs/doom/` |
| **BroomWorm** | drive a floor robot: cover a room, dodge furniture, come back to the dock | `src/broomworm/` | `docs/broom/` |

Everything they share — the connectome, the leaky-integrate-and-fire simulator,
the sensory and motor adapters, the body contract, evolution, CMA-ES, the
search and the benchmark — lives once, in `src/wormlab/`.

```bash
make tracks     # what lives where
make setup      # venv, the connectome, the Doom data
make check      # ruff, mypy, the whole suite (both tracks)
make help       # every target; the robot's are prefixed broom-
```

## Why one repository

Because the two tracks keep answering each other's questions. The Doom track
found that the trigger it was using sits in the pharyngeal island, reachable
only through a single pair of neurons — a fact about the wiring that the robot
track would otherwise have had to discover for itself. The robot track measured
how the worm transfers to sensors it was never trained on, which is the same
question a new Doom level asks.

So findings that hold *whatever the task is* are written once, in
`docs/knowledge/`, with the numbers and the run they came from:

| File | What it holds |
|---|---|
| `docs/knowledge/connectome.md` | what the wiring itself allows and forbids |
| `docs/knowledge/search.md` | how to find weights in a fixed topology |
| `docs/knowledge/sensing.md` | how to put a world in front of 302 neurons |
| `docs/knowledge/method.md` | how to run an experiment without fooling yourself |

A track's own story stays in its own `findings.md`. See
`docs/knowledge/README.md` for the rule that keeps them apart.

## The Doom track

```bash
make demo-b1 LEVEL=doom4           # the hand-written floor on a simulator level
make demo-b4 LEVEL=doom4           # the same level inside the Doom engine
make demo-classic CLASSIC=e1m1     # the classic game's own first map
make watch-doomguy                 # watch it play, in a real Doom window
make evolve-doom CANDIDATE=worm    # train one brain
make lane-bakeoff                  # the full three-seed bake-off (hours, resumable)
```

Where it stands, after stage 25 gave every row a denominator: an episode can
reach 22-46 points, and **no trained brain in this project is distinguishable
from flailing at random** (+0.37 to −24.52 by level). The hand-written floor is
the only thing that ever leaves that band — 59 % of what `doom6` allows. The
reasons are measured, not guessed — `docs/doom/findings.md`.

## The robot track

This track has its own command, `broomworm`, because these describe a *machine*
rather than a brain. In order, from the bench to the floor:

```bash
make broom-firmware                # compile the ESP32 firmware
make broom-motor-map LINK=fake     # which shift-register bit turns which wheel
broomworm selftest --link tcp      # day one: protocol, sensors, wheels
broomworm calibrate --link tcp     # metres per unit, and the wheel base
broomworm drive --teleop --record run.jsonl    # drive it, keep the log
broomworm compare-log --log run.jsonl          # replay that log in the simulator
broomworm robustness --brain X     # sweep the sensor preset's assumed numbers
```

Everything about brains, training and the benchmark stays on `wormlab`.

Where it stands: the simulation side is measured through stage 22 (transfer to a
new sensor preset, robustness sweeps, the cost of training), the firmware
compiles and the bench answers — `docs/broom/findings.md`,
`docs/broom/hardware.md`.

## The rules both tracks keep

- The connectome's **topology is fixed**; only synaptic weights are trained.
- Leaky integrate-and-fire neurons, graded activity, no body model.
- **No CNN, transformer or RL framework between the world and the brain.**
- No task logic in the brain, and no framebuffer until a vision stage.
- Deterministic seeds. A published row replays from what is committed.
- A design choice is settled by measurement, and the losing measurements are
  written down too.
