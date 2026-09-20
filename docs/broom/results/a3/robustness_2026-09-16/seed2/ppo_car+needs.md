# ppo_car+needs around the car preset

| parameter | value | reward | vs base | survived | collisions | coverage |
|---|---|---|---|---|---|---|
| base | (preset) | 44.4 ± 19.3 | +0.0 | 0.83 | 17.8 | 0.24 |
| noise_sigma | 0.15 | 45.9 ± 19.6 | +1.5 | 0.92 | 21.8 | 0.24 |
| noise_sigma | 0.3 | 40.3 ± 26.2 | -4.1 | 0.58 | 24.2 | 0.23 |
| dropout | 0.15 | 49.2 ± 28.3 | +4.8 | 0.83 | 31.3 | 0.28 |
| dropout | 0.3 | 47.3 ± 25.4 | +2.9 | 0.75 | 31.8 | 0.27 |
| odom_sigma | 0.2 | 45.3 ± 28.1 | +0.9 | 0.75 | 29.3 | 0.26 |
| odom_sigma | 0.4 | 39.7 ± 25.9 | -4.7 | 0.58 | 33.4 | 0.23 |
| delay | 2 | 38.8 ± 15.9 | -5.6 | 0.75 | 14.8 | 0.22 |
| delay | 3 | 28.8 ± 20.1 | -15.6 | 0.42 | 16.5 | 0.21 |
| beacon_fov | 20 deg | 40.4 ± 20.7 | -4.0 | 0.83 | 41.8 | 0.22 |
| beacon_fov | 15 deg | 43.5 ± 20.7 | -0.9 | 0.83 | 21.9 | 0.23 |
| beacon_range | 3 | 39.9 ± 20.9 | -4.5 | 0.75 | 25.6 | 0.22 |
| beacon_range | 2 | 35.5 ± 23.8 | -8.9 | 0.67 | 31.4 | 0.21 |

largest drops: delay=3, beacon_range=2, delay=2
