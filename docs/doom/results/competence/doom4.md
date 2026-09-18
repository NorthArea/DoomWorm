# Competence map — `doom4`

3600 decisions over 6 unseen maps, 6 proposals each,
rolled 12 ticks forward, replanning every tick. Mean episode reward
3.13.

**brain already right** is how often the connectome's own first answer was the one
the rollout kept. **what search added** is the score it gained over that answer.
**how far it moved** is the distance between the two intents.

| situation | decisions | brain already right | what search added | how far it moved |
|---|---|---|---|---|
| wall ahead | 116 | 51 % | +0.06 | 0.49 |
| monster on the gun line | 122 | 44 % | +0.13 | 0.60 |
| monster off to one side | 551 | 44 % | +0.13 | 0.54 |
| exit ahead | 388 | 43 % | +0.07 | 0.56 |
| exit to one side | 2423 | 42 % | +0.07 | 0.58 |
