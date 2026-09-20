# ppo_car+needs around the car preset

| parameter | value | reward | vs base | survived | collisions | coverage |
|---|---|---|---|---|---|---|
| base | (preset) | 42.0 ± 17.7 | +0.0 | 1.00 | 43.6 | 0.23 |
| noise_sigma | 0.15 | 34.1 ± 23.7 | -7.9 | 0.83 | 46.7 | 0.21 |
| noise_sigma | 0.3 | 36.2 ± 19.4 | -5.9 | 0.92 | 28.9 | 0.20 |
| dropout | 0.15 | 42.1 ± 15.4 | +0.1 | 0.92 | 30.2 | 0.23 |
| dropout | 0.3 | 43.6 ± 18.5 | +1.5 | 0.92 | 33.0 | 0.24 |
| odom_sigma | 0.2 | 46.0 ± 15.7 | +4.0 | 1.00 | 25.9 | 0.23 |
| odom_sigma | 0.4 | 44.7 ± 21.4 | +2.6 | 0.75 | 12.2 | 0.24 |
| delay | 2 | 39.0 ± 18.6 | -3.0 | 0.92 | 27.2 | 0.20 |
| delay | 3 | 33.1 ± 22.7 | -9.0 | 0.67 | 19.3 | 0.20 |
| beacon_fov | 20 deg | 49.9 ± 19.2 | +7.9 | 1.00 | 13.2 | 0.24 |
| beacon_fov | 15 deg | 51.8 ± 13.6 | +9.7 | 1.00 | 17.8 | 0.23 |
| beacon_range | 3 | 44.2 ± 20.2 | +2.2 | 0.92 | 16.8 | 0.22 |
| beacon_range | 2 | 32.7 ± 17.8 | -9.3 | 0.67 | 19.5 | 0.21 |

largest drops: beacon_range=2, delay=3, noise_sigma=0.15
