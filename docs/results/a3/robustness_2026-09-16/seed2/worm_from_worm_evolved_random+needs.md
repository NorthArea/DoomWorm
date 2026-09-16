# worm_from_worm_evolved_random+needs around the car preset

| parameter | value | reward | vs base | survived | collisions | coverage |
|---|---|---|---|---|---|---|
| base | (preset) | 37.8 ± 12.8 | +0.0 | 0.92 | 10.5 | 0.20 |
| noise_sigma | 0.15 | 35.6 ± 16.3 | -2.2 | 0.92 | 10.1 | 0.18 |
| noise_sigma | 0.3 | 40.6 ± 8.7 | +2.8 | 1.00 | 5.2 | 0.18 |
| dropout | 0.15 | 39.7 ± 9.7 | +1.9 | 1.00 | 16.2 | 0.19 |
| dropout | 0.3 | 33.9 ± 11.9 | -3.9 | 1.00 | 24.8 | 0.19 |
| odom_sigma | 0.2 | 38.8 ± 12.1 | +1.0 | 0.92 | 12.4 | 0.20 |
| odom_sigma | 0.4 | 22.5 ± 15.3 | -15.3 | 0.75 | 13.3 | 0.15 |
| delay | 2 | 32.3 ± 12.4 | -5.5 | 0.92 | 12.5 | 0.17 |
| delay | 3 | 34.7 ± 14.4 | -3.1 | 0.92 | 14.6 | 0.17 |
| beacon_fov | 20 deg | 38.1 ± 11.0 | +0.3 | 0.92 | 9.7 | 0.19 |
| beacon_fov | 15 deg | 29.9 ± 16.8 | -7.9 | 0.75 | 10.5 | 0.18 |
| beacon_range | 3 | 36.2 ± 17.5 | -1.6 | 0.75 | 9.9 | 0.21 |
| beacon_range | 2 | 29.1 ± 16.0 | -8.6 | 0.67 | 9.0 | 0.17 |

largest drops: odom_sigma=0.4, beacon_range=2, beacon_fov=15 deg
