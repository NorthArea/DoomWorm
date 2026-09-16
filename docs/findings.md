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
| C5 robustness to wrong sensor numbers | sweep of noise, dropout, odometry drift, delay, camera view and range around the car preset | _running_ | _running_ | | | `results/a3/` (stage 23.1) |
| C6 cost of training | reward per generation / step; number of training maps (12 maps x 2 noise repeats retraining running) | _running_ | _running_ | | | stage 23.2 |
| C7 hybrids | worm as the motivational layer over a net, worm + small net | | | | todo | stage 23.3 |
| C8 reality gap | the car: same saved brain in the simulator and in the room | | | | needs the machine | stage 22.2+ |
| C9 another environment | Doom (track B) | | | | on the owner's word | |

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

Open questions the running axes answer: does the worm degrade more slowly
than PPO when the assumed sensor numbers are wrong (C5); does more training
data close the worm's retraining gap or PPO's transfer gap (C6).
