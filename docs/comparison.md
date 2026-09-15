# Stage 11: real vs random vs shuffled vs free topology

Plan §17 / §49. Same 302 neurons, same sensory and motor mappings (named
neurons), same tonic AVB drive, same per-target weight normalisation, same
world seeds (train 0-2, held-out 1000-1004, fixed obstacle, 2 food with
respawn, 400 steps), same evolution budget (population 40, 25 generations,
sigma 0.02, weights in [-1, 1], evolution seed 0). Only the wiring differs.

Raw files: `docs/results/compare_2026-09-15/` (summary, learning curves,
per-variant results). Reproduce with `make compare` (about 15 min with the
four variants in parallel).

## Result (one evolution run per variant)

| variant | connections | train best | gens to >0 | held-out fitness | food | collisions | survival | distance |
|---|---|---|---|---|---|---|---|---|
| real | 5905 | 55.6 | 3 | 29.1 | 14 | 0 | 370 | 51.8 |
| random | 5905 | 54.0 | 8 | 26.7 | 18 | 185 | 383 | 43.3 |
| shuffled | 5905 | 79.9 | 6 | 42.7 | 24 | 175 | 380 | 49.5 |
| dense (free) | 3709 non-zero of 91204 | 71.9 | 1 | 35.8 | 22 | 353 | 384 | 45.7 |

Held-out columns are sums (food, collisions) or means (fitness, survival
ticks, distance) over the 5 unseen seeds.

Learning curves (best fitness on train seeds):

| generation | real | random | shuffled | dense |
|---|---|---|---|---|
| 0 | -18 | -19 | -19 | -9 |
| 5 | 27 | -8 | -2 | 50 |
| 10 | 40 | 14 | 55 | 62 |
| 15 | 49 | 37 | 80 | 62 |
| 24 | 56 | 54 | 80 | 72 |

## Reading

1. **Every topology learns the task.** All four go from about -19 to +54 or
   better in 25 generations. Weights, not wiring, carry most of the result.
2. **Real ≈ Random on final held-out fitness** (29 vs 27). The biological
   wiring is not a strong prior for this task at this budget.
3. **Real learns fastest among the sparse variants** (positive fitness at
   generation 3 vs 6 and 8) and is the **only variant with zero collisions**
   on unseen maps. Its reflexes (touch -> slow down, stage 8-9) survive
   training; the controls instead learn to eat more while bumping into
   things, because the collision penalty is capped per episode.
4. **Shuffled > Real** on eating (24 vs 14 food). Degree-preserving rewiring
   keeps the hub structure (AVA/AVB/AVD as hubs) but frees the paths; one
   run cannot say whether that is signal or luck.
5. **Free (dense) is the fastest learner** (positive at generation 1) and
   ends between shuffled and real; it also collides the most.

## Caveats

- n = 1 evolution run per variant. Differences of ~10 fitness are within the
  spread seen across seeds in stage 10. A claim about ordering needs at
  least 3 evolution seeds per variant (`--seed`), which is a 45-minute job.
- The task is easy and the reward is generous: 25 generations saturate it.
  Harder maps (stage 12, track A) may separate the variants more.
- All variants share the artificial parts: L/R mapping, tonic AVB, gains.

## Decision (Plan §17.1)

Outcome class: **Real ~= Random**, with two real advantages that matter for
a robot (learns faster early, drives without collisions) and one
disadvantage (eats less).

Decision: **continue with the C. elegans connectome as the primary brain**,
because (a) the project's question is about it, (b) its early learning and
collision-free behaviour are the properties the reactive layer of the
robot needs, (c) the cost of keeping controls is small. **Random, shuffled
and dense stay as mandatory controls** in every later training run (track
A stages 12, 15-17) so the comparison is repeated on harder tasks and with
noisy sensors, where the plan expects the motifs to matter more or less.
Revisit at the end of stage 17; if Real is still not ahead on any axis,
switch the robot brain to the Free mode and keep the worm as the control.
