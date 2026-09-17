# Findings: where the natural network wins, ties or loses

The project's result (Plan §0, §48). One row per axis of the conditions
matrix (Plan §20.7): the connectome worm against the best network trained
from scratch under the same engineered layer, budget and benchmark. Numbers
are mean reward over >= 3 training seeds unless marked; the "profit" column
is worm minus best self-trained, in reward units. Verdict: **win** / **tie**
(inside the spread) / **loss**.

| Axis | Condition | Worm | Best self-trained | Profit | Verdict | Rows |
|---|---|---|---|---|---|---|
| C1 same task, same budget | vacuum preset, apartment/clean, evolution pop 40 x 25 gen (PPO 300k steps) | real 32.1 ± 5.8 | PPO 61.9 ± 0.2; small rnn 34.5 ± 6.4 | −29.8 vs PPO; −2.4 vs rnn | **loss** to PPO, **tie** with the same-size rnn | `results/a2/a2_bakeoff_2026-09-15.md` |
| C1 controls | same, topology controls | real 32.1 | random 15.9 ± 4.1; shuffled 40.4 ± 9.7; dense 26.2 ± 3.1 | +16.2 vs random | real > random, real ≈ shuffled | same |
| C2 built-in motives (curriculum) | worm pre-trained on the food task, then clean, same total budget per stage | curriculum 44.0 ± 7.6 | rnn 34.5 ± 6.4 (no curriculum available for it: it has no food-task ancestor) | +9.5 | **win** over the same-size net; still −17.9 vs PPO | same |
| C3 transfer without retraining | vacuum-trained brains run on the car preset (3 rays instead of 5, no bumper, dead reckoning, camera marker) | curriculum 41.2 ± 0.2 (−6 %); real 34.3 ± 3.2 (+7 %) | PPO 27.3 ± 3.6 (−56 %); rnn 27.2 ± 6.5 (−21 %) | +13.9 vs PPO | **win**: the worm's sector inputs survive the sensor change, the nets' input vectors do not | `results/a3/stage22_car_2026-09-15.md` §2 |
| C4 retraining on the new sensors | car preset, same budget as A2 | curriculum retrained 30.1 ± 5.4 (worse than transferred: overfits 3 training maps) | PPO retrained 43.0 ± 4.0; rnn 30.1 ± 5.0 | −1.8 (transferred worm 41.2 vs PPO 43.0) | **tie**; the worm has half the collisions (14 vs 31) | same §2 |
| C4 under layer v2 | car preset, marker search in the layer | curriculum (transferred) 42.1 ± 3.7 | PPO-car 46.5 ± 7.7 | −4.4 | **tie** (spreads overlap); worm 17 vs 26 collisions | same §3 |
| C5 robustness to wrong sensor numbers | one car-preset number at a time made worse (noise x3/x6, dropout x3/x6, odometry drift x2/x4, delay 2-3 ticks, camera view 20/15 deg, marker range 3/2 u); 12 episodes per cell, 3 seeds | curriculum (transferred) base 41.3 ± 3.2; worst cell −12.3 (marker range 2 u), −11.6 (drift x4), −11.3 (delay 3); noise x6 only −2.2 | PPO-car base 47.8 ± 6.5; worst −18.8 (delay 3), −11.7 (marker range 2), −10.5 (noise x6); drift x4 only −5.7 | worst-case drop −30 % (worm) vs −39 % (PPO) | **tie**, split by parameter: the worm degrades less under noise amplitude and latency, PPO less under odometry drift and a narrower camera; both fall the same when the marker range shrinks (that is the layer, not the brain) | `results/a3/robustness_2026-09-16/summary_3seeds.md` |
| C6a more training data, same budget | car preset, retraining on 12 maps x 2 noise repeats instead of 3 x 1; evolution pop 40 x 25 gen, PPO 300k steps; layer v2 | curriculum worm 44.4 ± 5.6 (was 30.1 on 3 maps; transferred 42.1) | PPO 37.5 ± 8.4 (was 46.5 on 3 maps) | +6.9 | **tie leaning worm** (spreads 5.6 / 8.4 overlap): more maps cure the worm's overfit; PPO at a fixed 300k steps gets fewer steps per map and loses | `results/a3/benchmark_car12_2026-09-16/summary_3seeds.md` |
| C6b cost in environment steps | what each budget above actually consumes | evolution: 40 genomes x 25 gen x 24 episodes = 24 000 episodes = 19 M env steps | PPO: 300 000 steps = 375 episodes | worm needs ~64x the experience | **loss** on sample efficiency: the natural wiring does not make evolution cheap; a gradient learner reaches the same reward on 1/64 of the data | same, `docs/brains/car_12maps/train.log` |
| C7 hybrids | the seed's C6 curriculum worm frozen as a reflex module, its wheels and trigger three extra channels of a small net from scratch; only the net trained. Car preset, layer v2, 12 maps x 2 repeats, 3 seeds -- the C6 protocol | hybrid 43.2 ± 1.2; the worm alone (C6) 44.4 ± 5.6 | the same net without the worm 37.7 ± 1.4; PPO (C6) 37.5 ± 8.4 | +5.5 over the bare net | **the worm helps as a part, and only up to itself**: bolted under a from-scratch net it lifts that net by 5.5 points, exactly to the worm's own level, with no synergy beyond it. The hybrid is the steadiest row in the project (± 1.2 against the worm's ± 5.6) | `runs/c7/benchmark/leaderboard.md`, brains `runs/c7/seed*/` |
| C8 reality gap | the car: same saved brain in the simulator and in the room | | | | needs the machine | stage 22.2+ |
| C9a another environment, same task and budget | mini-Doom `doom4` (gun, one enemy, exit), no planner, ideal sensors, evolution 40 x 25 on 3 maps / PPO 300k, 12 unseen maps, 3 seeds | real −13.7 ± 5.2; curriculum −8.1 ± 4.0 | PPO −3.3 ± 6.5; rnn −8.5 ± 1.7; ncp −9.5 ± 1.1 | −10.4 (curriculum −4.8) | **loss**; and the whole field loses to the zero-learning `doomguy` floor (+4.8), so at the A2 budget nobody learned Doom | `results/b/b2_doom_2026-09-16.md` |
| C9a controls | same | real −13.7 | random −4.9 ± 6.2; shuffled −2.1 ± 6.8 | −11.6 vs shuffled | **loss to its own controls**: on Doom the connectome is worse than the same topology shuffled (on the vacuum it was a tie) | same |
| C9b harder level | `doom6` (apartment, two walking enemies, gun), same protocol | real −40.1 ± 3.3; curriculum −24.6 ± 1.0 | ncp −21.3 ± 2.3; PPO −33.0 ± 6.4; rnn −46.2 ± 4.0 | −18.8 (curriculum −3.3) | **loss**; floor `doomguy` +27.2 with 1.33 kills and 75 % exits | same |
| C9c the Doom engine, same brains unchanged | the same seeded layouts run by ViZDoom (`vizdoom1`..`vizdoom6`), no retraining | curriculum 7.1 ± 3.5 — the best trained brain on `vizdoom4` | PPO 0.8 ± 2.6; rnn −1.1 ± 0.9; ncp −2.2 ± 0.5 | +6.3 vs PPO | **win among trained brains** (still far under the floor's +27.4): the engine reverses the simulator's order, the worm family on top | same, `runs/benchmark_doom/train_doom4/*/eval_vizdoom*` |
| C9d a map nobody here drew | stage B11, the stock ViZDoom scenario `defend_the_center` (the engine's own geometry and monsters), brains trained on `doom4`/seed 0 run unchanged, 4 episodes each | real 3.2 ± 16.8 with 2.0 kills on 6.5 shots; curriculum −32.4 with **no shot fired** | rnn −29.6 and PPO −37.3, neither ever fires; shuffled 14.9 with 4.0 kills on 26 shots | +32.8 vs PPO, −19.6 vs the floor | **win over the trained nets, loss to the floor** (`doomguy` 22.8, 2.8 kills on 2.8 shots, survives half): on a map outside our generator only the worm family pulls the trigger at all. Four episodes and one training seed -- a direction, not a measurement | `runs/benchmark_doom/stock/leaderboard.md` |
| C9 caveat on every shooting row | measured on untrained brains, `doom4` | real pulls the trigger in **0.0 %** of ticks | shuffled 77.0 %, random 34.8 % | — | **the rows below compare our motor mapping, not the wiring**: the trigger group M3/M4/MC sits in the pharyngeal island, which the real connectome joins to the rest through 5 connections via RIPL/RIPR only, so no sensor can reach it; the shuffle gives it 399 crossing connections and direct input from ADL (`aim`) and AVA/AVB. To be redone with a reachable trigger (stage B2c) | `docs/assumptions.md` 2026-09-16 |
| C9 shooting | what the trigger actually does, `doom4` | real 0.00 kills on 3.1 shots; curriculum 0.03 on 29.9 | PPO 0.00 on 11.2; rnn / ncp 0.00 on 0.0 (never pull the trigger) | floor: 0.50 kills on 5.7 shots | **loss**: the evolved brains spray or never fire; aiming is not found at this budget, though `aim` makes it a few-synapse function | same |

## What the map says so far (2026-09-16)

- On the task it was trained for, with the same budget, the worm does not
  beat a network trained from scratch: it ties a same-size recurrent net and
  loses clearly to PPO (C1). Its wiring is better than random wiring and no
  better than shuffled wiring (C1 controls).
- The worm's advantages appear off the training distribution. Pre-training on
  a simpler task carries over (C2, +9.5 over the same-size net), and a
  change of sensors that the brain never saw costs it 6 % where PPO loses
  56 % (C3). Retrained on the new sensors PPO catches up (C4), so the
  worm's profit is in **not having to retrain**, plus fewer collisions.
- Where the difference is decided is the engineered layer, not the brain:
  moving docking and the bumper reflex into the layer raised every brain
  (A2 21.8-21.9, A3 22.1e) and let the worm-vs-random ordering appear.

- Training data (C6): given more maps at the same protocol budget the worm
  recovers from its overfit and edges past PPO (44.4 vs 37.5), but it pays
  ~64x the environment steps for it. The worm's profit is never in cheap
  training; it is in what it carries over without training.
- Wrong sensor numbers (C5) do not separate the two: the worm is the calmer
  one under noise amplitude and latency, PPO under odometry drift and a
  narrower camera. Neither is a robustness win; the layer's marker range is
  the shared weak point.

- Doom (C9) is the first axis where the worm loses to its own controls, not
  just to a network: shuffled beats real by 11.6 points on `doom4`. The task
  asks for something the connectome's wiring gives no head start on -- hold a
  heading, pass a doorway, wait for a target to cross the gun line -- and at
  the A2 budget no candidate of any kind beats the hand-written floor.
- The same brains score higher in the Doom engine than in our mini-Doom
  (curriculum worm `doom6` −28.3 -> `vizdoom6` +13.0). Our enemies damage
  every tick they see you, Doom's zombies fire in bursts with animation
  delays; the simulator is the harsher of the two, which is the safe
  direction (Plan §2.4) but makes every simulator row pessimistic.

- Hybrids (C7): the connectome is worth something as a *component* -- it carries a
  from-scratch net from 37.7 to 43.2 -- but the ceiling is the worm's own number.
  What the hybrid buys on top is consistency across seeds, not reward.

Remaining axes: the real car (C8).
