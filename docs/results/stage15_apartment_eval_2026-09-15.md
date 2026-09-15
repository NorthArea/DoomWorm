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
