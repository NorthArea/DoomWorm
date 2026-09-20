# Stage 15: apartment maps (2x2 rooms, doors, one furniture piece per room)

Six unseen apartment seeds (3000-3005), 400-step cap. `rooms` = distinct rooms
the trajectory passed through (Plan §20.2 acceptance: reach a neighbouring
room through a door on an unseen map).

## Stage-12 brain (trained on random circular-obstacle maps), no apartment training

| seed | ticks | distance | food | rooms | collisions | reward |
|---|---|---|---|---|---|---|
| 3000 | 250 | 16.8 | 0 | 1 | 24 | -29.9 |
| 3001 | 250 | 13.9 | 0 | 2 | 93 | -38.6 |
| 3002 | 400 | 35.4 | 3 | 3 | 0 | 34.5 |
| 3003 | 250 | 10.3 | 0 | 1 | 117 | -38.6 |
| 3004 | 250 | 3.0 | 0 | 1 | 198 | -39.5 |
| 3005 | 339 | 21.8 | 2 | 1 | 84 | -17.2 |

Doors are passed on 2 of 6 maps without any apartment training. Flat walls
are new to this brain: its curved sweeping strategy slides along them and
racks up collisions, and on 4 of 6 maps it starves in its first room.

## Curriculum training on apartments (25 generations from the stage-12 brain)

Training on 4 apartment seeds stalled: best -18 -> -2 by generation 5, flat
afterwards. On the same six unseen apartments:

| brain | food | rooms per map | maps with >= 2 rooms | collisions | mean reward | mean ticks |
|---|---|---|---|---|---|---|
| stage 12 (no apartment training) | 5 | 1, 2, 3, 1, 1, 1 | 2 / 6 | 516 | -21.6 | 290 |
| stage 15 (apartment-trained) | 0 | 1, 2, 1, 1, 1, 1 | 1 / 6 | 141 | -28.2 | 250 |

The apartment-trained brain learned to collide less by moving less and
finds no food; it is worse at the actual task. Stage 15 acceptance (a
door passed on an unseen map) holds for the stage-12 brain on 2 of 6 maps.
Conclusion: with the current sensors (three rays, no bumper, no wall
sensor), the current mapping and evolution at sigma 0.02, the worm has hit
a ceiling on structured maps. This is the input to the platform rebuild
(Plan §20.3) and the brain bake-off (Plan §20.4).
