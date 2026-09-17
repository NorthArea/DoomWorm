# Stages

Each row: what it is, the tests that hold it, the demo that runs it, and where
its numbers live. Nothing starts before the row above it is green.

The project used to carry two tracks — a home robot and a Doom player — on one
platform. Only the Doom player is left; the vacuum, the kit car and everything
built for them live in the git history up to `1bf247a`.

## Platform

| # | Stage | Tests | Demo | Status |
|---|-------|-------|------|--------|
| 1 | 2D world, connectome loader, LIF simulator, sensory and motor adapters | `tests/test_simple_2d.py`, `tests/test_brain.py`, `tests/test_adapters.py` | `make demo-0`, `make demo-1` | done |
| 2 | Evolution on the synaptic weights; the food task as the curriculum world | `tests/test_evolution.py`, `tests/test_evolve_worm.py` | `make train-worm` | done |
| 3 | Topology controls (random, shuffled, dense) and the comparison that gates them | `tests/test_variants.py` | `make compare` | done: the connectome stays the primary brain; the controls are trained in every later run |
| 4 | The benchmark: seeded unseen maps, one protocol, a leaderboard | `tests/test_brain_interface.py` | `make benchmark` | done |

## Doom

| # | Stage | Tests | Demo | Status |
|---|-------|-------|------|--------|
| 5 | Doom on the 2D platform: monsters (line of sight, hurt in range, walk), a hitscan gun on a 50-round magazine, an exit that ends the episode, six seeded levels `doom1`..`doom6`, the `aim` and `ammo` channels, hit/kill reward, a trigger for every candidate, the hand-written `doomguy` floor | `tests/test_doom.py` | `make demo-b1 LEVEL=doom4` | done 2026-09-16 |
| 6 | The Doom engine behind the same world contract: the seeded layout written as a PWAD, ViZDoom in PLAYER mode, structured observations from game variables and the object list (no framebuffer), wheels calibrated to the simulator's speed, levels `vizdoom1`..`vizdoom6` | `tests/test_vizdoom.py` (skipped without the `doom` group) | `make demo-b4 LEVEL=doom4` | done 2026-09-16 |
| 7 | Every candidate on the benchmark, three seeds, trained on `doom4` and `doom6`, evaluated on all six levels and, unchanged, in the engine | `tests/test_bakeoff.py`, `tests/test_doom.py` | `make evolve-doom CANDIDATE=worm LEVEL=doom4`, `make benchmark-doom` | done 2026-09-16: no trained brain beats the floor; on `doom4` shuffled −2.1 > PPO −3.3 > random −4.9 > curriculum worm −8.1 > rnn −8.5 > ncp −9.5 > worm −13.7, floor +4.8. Tables in `docs/results/b/b2_doom_2026-09-16.md` |
| 8 | A stock ViZDoom scenario — a map nobody here drew. The engine's blocking sector lines become the world's angled walls, its object list the enemies; sensing, `aim`, the trigger and the reward are the platform's own code. Levels `stock_defend`, `stock_corridor`, `stock_home` | `tests/test_stock.py` | `make demo-b11 STOCK=stock_defend`, `make watch-stock` | done 2026-09-16: floor 22.8 (2.8 kills) > shuffled 14.9 > worm 3.2 > follower 1.1 > rnn −29.6 > curriculum −32.4 > PPO −37.3. Only the worm family fires at all |
| 9 | A trigger the connectome can reach. The pharyngeal group M3/M4/MC is an island joined to the rest by five connections, all through RIP, so `aim` never arrived: untrained fire 0.134 with an enemy on the gun line against 0.157 with none. With `aim` delivered to RIP the same untrained worm answers 0.906 against 0.143 | `tests/test_doom.py` | — | done 2026-09-17 |
| 10 | A target the worm can turn toward. The gun is bolted to the chassis, so aiming is turning, and the monster only existed on the escape pathway (ASH). The `prey_*` channels read it as a lateralised attractant on CEPD — chosen by hits landed, not by the story: CEPD 20 hits / 27 shots against a standing target, PLM 12/22, the bare connectome 0. Narrowed on 2026-09-17: every one of those hits is head-on, and no pair at all hits from the side | `tests/test_doom.py` | — | done 2026-09-17. Untrained on `stock_defend`: −5.4 with 0.50 kills and 0.75 survival, against −40.9 and certain death before |
| 11 | The vehicle leaves the brain's contract: `Drive(forward, turn, strafe, fire)` plus a body that turns it into actuators | `tests/test_body.py` | — | done 2026-09-17, verified identical to ten decimals against the pre-change code |
| 12 | Re-measure the shooting rows with the reachable trigger and the prey channel | `tests/test_bakeoff.py` | `uv run scripts/doom_b2d_lane.py` | done 2026-09-17: on `doom4` the worm gained 10.9 and caught its controls; on `doom6` it still trails them and the curriculum worm lost 16.0; in the engine the fixes cost points. Aim improved everywhere (2-9x the controls' kills per shot). Rows and the diagnosis in `docs/findings.md` |
| 18 | Squeeze everything outside the brain: a genome cut to the interface, sep-CMA-ES, the adapter gains as genes | `tests/test_cmaes.py` | `uv run scripts/squeeze_lane.py`, `scripts/squeeze2_lane.py` | done 2026-09-17: CMA + interface **+0.66 ± 3.4** on `doom4`, the worm's first positive row; `doom6` −22.10 against −38.43. Gains as genes failed (−6.93). The ceiling is the architecture, not the optimiser |
| 13 | Reorientation: the worm hunts what is in front of it and cannot come round onto a target 40-90 degrees off the bow — measured, and not fixable by routing: 0 of 31 free sensory pairs land a single hit from the side. Left to evolution (stage 12), then to the pirouette, which needs a temporal derivative the LIF neurons lack | `tests/test_doom.py` | | todo, blocked on stage 12's numbers |
| 14 | Strafe: the engine has it, the animal does not. Whether the nets get a fourth output the worm cannot have is a question for the map | | | todo |
| 15 | Finish the strip: the world still carries the vacuum's dirt map, dock and battery as inert scaffolding, and `reward.py` their terms. Dead for Doom, so it is a cleanup rather than a measurement | `tests/test_simple_2d.py` | | todo |
| 16 | Vision: the framebuffer through a very small encoder into sensory channels (Plan §11) | | | todo, only after the structured rows close |
