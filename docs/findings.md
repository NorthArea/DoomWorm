# Findings: does the natural network play Doom better?

The project's result. One row per question: the connectome worm against the
best network trained from scratch, under the same budget and the same
benchmark. Numbers are mean reward over >= 3 training seeds unless marked; the
profit column is worm minus best self-trained, in reward units. Verdict:
**win** / **tie** (inside the spread) / **loss**.

Rows measured on the vacuum robot and the kit car, which this project used to
carry as a second track, are not here: they live in the git history up to
`1bf247a`.

| # | Condition | Worm | Best self-trained | Profit | Verdict | Rows |
|---|---|---|---|---|---|---|
| 1 same task, same budget | `doom4` (gun, one monster, exit), 3 training maps, evolution 40 x 25 / PPO 300k, 12 unseen maps, 3 seeds | real −13.7 ± 5.2; curriculum −8.1 ± 4.0 | PPO −3.3 ± 6.5; rnn −8.5 ± 1.7; ncp −9.5 ± 1.1 | −10.4 (curriculum −4.8) | **loss** — and the whole field loses to the zero-learning `doomguy` floor (+4.8), so at this budget nobody learned Doom | `results/b/b2_doom_2026-09-16.md` |
| 1 controls | same | real −13.7 | random −4.9 ± 6.2; shuffled −2.1 ± 6.8 | −11.6 vs shuffled | **loss to its own controls** — the first axis where the wiring is worse than the same wiring shuffled | same |
| 2 a harder level | `doom6` (apartment, two walking monsters, gun) | real −40.1 ± 3.3; curriculum −24.6 ± 1.0 | ncp −21.3 ± 2.3; PPO −33.0 ± 6.4; rnn −46.2 ± 4.0 | −18.8 (curriculum −3.3) | **loss**; the floor takes +27.2 with 1.33 kills and 75 % exits | same |
| 3 the real engine, same brains | the same seeded layouts run by ViZDoom, no retraining | curriculum 7.1 ± 3.5 — the best trained brain on `vizdoom4` | PPO 0.8 ± 2.6; rnn −1.1 ± 0.9; ncp −2.2 ± 0.5 | +6.3 | **win among trained brains** (still under the floor's +27.4): the engine reverses the simulator's order and the worm family comes out on top | same, `runs/benchmark_doom/train_doom4/*/eval_vizdoom*` |
| 4 a map nobody drew | the stock ViZDoom scenario `defend_the_center`, brains from `doom4`/seed 0, 4 episodes each | real 3.2 ± 16.8 with 2.0 kills; curriculum −32.4 with no shot fired | rnn −29.6 and PPO −37.3, neither ever fires; shuffled 14.9 with 4.0 kills | +32.8 vs PPO, −19.6 vs the floor | **win over the trained nets, loss to the floor**: outside our own generator only the worm family pulls the trigger at all. Four episodes, one seed — a direction, not a measurement | `runs/benchmark_doom/stock/leaderboard.md` |
| 5 hybrids | a frozen trained brain as a reflex module under a small net; only the net trained | hybrid 43.2 ± 1.2 | the same net alone 37.7 ± 1.4 | +5.5 over the bare net | **the worm helps as a part, and only up to itself**: it lifts a from-scratch net by 5.5 points, exactly to the worm's own level, and steadies it across seeds. Measured on the conditions of the old second track; to be re-run on Doom | `runs/c7/benchmark/leaderboard.md` |
| 6 shooting, as it stands | what the trigger actually does on `doom4` | real 0.00 kills on 3.1 shots; curriculum 0.03 on 29.9 | PPO 0.00 on 11.2; rnn and ncp never fire | floor: 0.50 kills on 5.7 shots | **superseded** — every row above was measured while the trigger was unreachable and the monster was only a source of pain. See the two fixes below; stage 12 re-measures them | same |

## What was wrong with the rows above

Two defects, both ours, both in the mappings rather than in the connectome, and
both found by measuring untrained brains:

- **The trigger was unreachable.** It sits on the pharyngeal pumping group
  M3/M4/MC, and the pharyngeal nervous system is an island: exactly five
  connections cross its boundary, all through RIPL/RIPR. Untrained, the real
  worm pulled the trigger in 0.0 % of ticks and answered 0.134 with a monster
  on the gun line against 0.157 with none — pure noise. Its shuffled control
  has 399 crossing connections after the rewiring and fired in 77 % of ticks.
  **That is why the control beat the connectome on every shooting row**, not
  better wiring. With `aim` delivered to RIP, the animal's own door, the same
  untrained worm answers 0.906 against 0.143.
- **There was nothing to turn toward.** The gun is bolted to the chassis, so
  aiming is turning, and the monster reached the brain only as `danger_*` into
  ASH — the escape pathway, which turns the body *away*. The `prey_*` channels
  now read it as a lateralised attractant on CEPD, the dopaminergic pair that
  in the animal reports food under the nose and slows it down.

With both in place and **no training at all**, the connectome on the stock map
went from never firing and dying in every episode to 0.50 kills and surviving
three quarters of them. Against a standing target it closes from 10 units dead
ahead and lands 20 hits; 60 degrees off the bow it still cannot come round.

## What the map says so far

- On the task it was trained for, at the A-track budget, the worm loses — to
  PPO, to a same-size net, and to its own shuffled control. The last one was an
  artefact of our motor mapping and is being re-measured.
- The worm's advantages show up **off the training distribution**: in the real
  engine, and on a map from someone else's generator, it is the only trained
  family that fights at all.
- Nothing trained has beaten the hand-written floor yet. That is the honest
  headline of the project so far.
- The connectome is worth something **as a component**: bolted under a
  from-scratch net it carries that net up to the worm's own level, and makes it
  far steadier across seeds — but not past it.
