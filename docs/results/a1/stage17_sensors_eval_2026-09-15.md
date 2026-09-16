# Stage 17: sensor suite, noise and delay

Stage-12 brain (trained with three ideal 45-degree rays), food task, six
unseen random maps (2000-2005), 400-step cap. Sums over maps, reward and
ticks are means.

| preset | rays | noise | dropout | delay | food | collisions | mean reward | mean ticks | distance |
|---|---|---|---|---|---|---|---|---|---|
| ideal | 3 at +-45, 0 | 0 | 0 | 0 | 6 | 0 | 0.1 | 318 | 177 |
| vacuum | 5 at +-60, +-30, 0 | 5% | 2% | 1 | 4 | 171 | -17.5 | 305 | 147 |
| noisy | 5 at +-60, +-30, 0 | 15% | 5% | 2 | 6 | 104 | -6.4 | 320 | 163 |

Reading: the loss comes mostly from the changed ray geometry (the legacy
"front" channel is now the single 0-degree ray, "left"/"right" the best of
30 and 60 degrees) and from latency, not from the noise level itself: the
noisier preset collides less than the moderate one. Brains must be trained
on the sensor preset they will run with; the ideal preset stays only as a
reference.
