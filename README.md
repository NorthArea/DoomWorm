# BroomWorm

A home vacuum robot whose brain is chosen on a fair test bed: the **C. elegans
connectome** (302 neurons, fixed wiring, trainable weights) against networks trained
from scratch (small RNN, PPO, NCP) and a Roomba-style controller, under identical
conditions: same simulated home, same sensor emulation, same engineered layer (map,
planner, needs), same training budget, one benchmark. The map of where the natural
network wins, ties or loses is **[docs/findings.md](docs/findings.md)**.

Two hardware phases (see [Plan.md](Plan.md)):

- **Phase A, the car**: ACEBOTT QD001 (ESP32, 4WD mecanum) + QD003 (K210 vision pack).
  Firmware, robot link, calibration and the `car` sensor preset are written; the machine
  has not been driven yet (stage 23.1 onwards).
- **Phase B, the vacuum**: a real vacuum platform with brush, cliff sensors and a
  charging dock.

```text
Environment -> sensor channels -> engineered layer -> Brain -> wheels
```

The simulation platform (stages 0-22.1, built as DoomWorm) is the shared foundation:
A1 platform and A2 bake-off are closed; on the vacuum benchmark PPO 61.9 > curriculum
worm 44.0 > shuffled 40.4 > RNN 34.5 > real worm 32.1 > Roomba 29.5. Stage progress:
[docs/stages.md](docs/stages.md). Every decision: [docs/assumptions.md](docs/assumptions.md).
The Doom track lives in a separate repository, `NorthArea/DoomWorm`.

## Requirements

- Python 3.12+ (project pinned to 3.14 in `.python-version`)
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync --all-groups          # add --group rl for the PPO / NCP candidates (torch)
uv run pre-commit install
make check                    # ruff, strict mypy, tests
```

## Everyday commands

```bash
make help                                                    # every shortcut
make benchmark-a2                                            # A2 bake-off on the vacuum preset
make benchmark-car                                           # the same brains on the car preset
make evolve-car CANDIDATE=worm                               # retrain a candidate for the car
make demo-22                                                 # drive over the robot link (simulator), replay the log
make selftest-sim                                            # rehearse day one: self-test, calibration, room drive
uv run broomworm selftest --link tcp                         # on the real car (phase A)
uv run broomworm drive --link tcp --teleop --record runs/x   # teleop with sensor recording
uv run broomworm benchmark --brain docs/brains/car/worm_curriculum.json --planner needs
```

Trained brains live in `docs/brains/`; `runs/` is git-ignored scratch.

## Layout

```text
src/broomworm/
  brain/           neuron model: Neuron, Synapse, Network, Simulator (LIF, graded)
  connectome/      C. elegans loader, internal graph, name mappings, topology controls
  adapters/        sensory (channels -> currents), motor (activity -> wheels)
  environments/    simple_2d world, maze/apartment/room maps, sensor suite + presets
                   (ideal | vacuum | noisy | car), worlds.py factories, gym_env.py
  layer/           the engineered layer outside the brain: occupancy grid, path and
                   coverage planning, needs arbitration, PlannerLayer, GradientFollower
  candidates/      the brains compared on the benchmark: worm, rnn, ncp, roomba, hybrid,
                   Brain / Trainable protocols, registry + load_candidate (PPO via rl group)
  learning/        reward, evolution, benchmark + leaderboard, bake-off, PPO, summary
  hardware/        robot link (sim | tcp), units, wire protocol, teleop, drive log,
                   sim-vs-real compare, room files, selftest, calibrate, plot
  experiments/     stage demos and training entry points (stages 0-14)
  visualization/   brain rasters, debug screen
  episode.py       the closed loop for any brain;  cli.py  the broomworm command
firmware/esp32_car/  ESP32 sketch speaking the robot protocol (not compiled yet)
data/connectome/     Cook 2019 connectome CSVs;  data/rooms/  real rooms
docs/                stages, assumptions, hardware contract, findings, results, brains,
                     history/plan_v1_doomworm.md (the previous plan, referenced by reports)
tests/               one module per layer
```

## Rules of the road

1. Stages in the order of Plan §6 then §7, one new complexity at a time.
2. A stage is done only when it works, is tested, and has a runnable demo.
3. No CNN/Transformer between the environment and the brain; adapters are simple transforms.
4. Connectome topology is fixed; only synaptic weights are trained.
5. No brain is compared outside the benchmark; deterministic seeds everywhere.

## License

MIT
