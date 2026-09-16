# worm_from_worm_evolved_random+needs around the car preset

| parameter | value | reward | vs base | survived | collisions | coverage |
|---|---|---|---|---|---|---|
| base | (preset) | 40.5 ± 24.8 | +0.0 | 0.83 | 21.0 | 0.24 |
| noise_sigma | 0.15 | 36.6 ± 32.8 | -3.9 | 0.67 | 18.3 | 0.22 |
| noise_sigma | 0.3 | 35.6 ± 23.9 | -4.9 | 0.75 | 16.6 | 0.20 |
| dropout | 0.15 | 34.9 ± 18.2 | -5.7 | 0.75 | 24.2 | 0.22 |
| dropout | 0.3 | 39.3 ± 16.2 | -1.2 | 0.83 | 34.3 | 0.23 |
| odom_sigma | 0.2 | 34.8 ± 20.3 | -5.8 | 0.58 | 19.8 | 0.23 |
| odom_sigma | 0.4 | 33.0 ± 21.1 | -7.5 | 0.58 | 19.1 | 0.22 |
| delay | 2 | 44.6 ± 13.5 | +4.1 | 0.92 | 24.0 | 0.26 |
| delay | 3 | 32.3 ± 22.7 | -8.2 | 0.67 | 24.2 | 0.21 |
| beacon_fov | 20 deg | 47.4 ± 9.6 | +6.9 | 1.00 | 25.0 | 0.26 |
| beacon_fov | 15 deg | 41.1 ± 14.7 | +0.6 | 0.83 | 28.8 | 0.25 |
| beacon_range | 3 | 41.3 ± 20.8 | +0.8 | 0.83 | 21.0 | 0.24 |
| beacon_range | 2 | 26.3 ± 22.7 | -14.3 | 0.42 | 14.7 | 0.20 |

largest drops: beacon_range=2, delay=3, odom_sigma=0.4
