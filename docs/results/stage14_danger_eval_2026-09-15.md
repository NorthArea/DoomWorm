# Stage 14: target + danger zone

Task: reach a respawning target while avoiding one danger zone (radius 1.0,
0.25 health per tick inside, death at 0). Six unseen random maps
(seeds 2000-2005), 400-step cap; without food, hunger ends every episode at
250 ticks. Sums over maps, reward and ticks are means.

| brain | targets | damage ticks | deaths | collisions | mean reward | mean ticks |
|---|---|---|---|---|---|---|
| stage 12 (no danger training) | 1 | 14 | 3 | 0 | -20.7 | 167 |
| stage 14 (25 gens on target + danger, from stage 12) | 2 | 0 | 0 | 208 | -21.1 | 250 |

Training (pop 40, 4 train maps): best -15 -> -2 on train maps. The trained
brain never enters the zone and never dies, at the price of hugging walls
(collisions), and it still reaches few targets. Danger avoidance learned;
target attraction weak: the only attractive signal it has is the same
chemosensory channel as food, and a single target gives one weak gradient.
