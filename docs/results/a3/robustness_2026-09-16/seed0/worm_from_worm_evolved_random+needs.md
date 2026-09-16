# worm_from_worm_evolved_random+needs around the car preset

| parameter | value | reward | vs base | survived | collisions | coverage |
|---|---|---|---|---|---|---|
| base | (preset) | 45.5 ± 8.3 | +0.0 | 1.00 | 13.3 | 0.23 |
| noise_sigma | 0.15 | 46.3 ± 13.7 | +0.8 | 1.00 | 12.8 | 0.22 |
| noise_sigma | 0.3 | 40.9 ± 7.6 | -4.5 | 1.00 | 7.8 | 0.19 |
| dropout | 0.15 | 33.3 ± 15.0 | -12.1 | 0.92 | 30.6 | 0.20 |
| dropout | 0.3 | 36.3 ± 13.6 | -9.2 | 0.92 | 31.9 | 0.22 |
| odom_sigma | 0.2 | 34.7 ± 13.4 | -10.7 | 0.75 | 18.2 | 0.21 |
| odom_sigma | 0.4 | 33.6 ± 14.7 | -11.9 | 0.75 | 11.8 | 0.20 |
| delay | 2 | 35.2 ± 19.9 | -10.3 | 0.83 | 16.2 | 0.20 |
| delay | 3 | 22.9 ± 14.9 | -22.6 | 0.83 | 25.8 | 0.16 |
| beacon_fov | 20 deg | 34.2 ± 16.0 | -11.3 | 0.67 | 12.2 | 0.21 |
| beacon_fov | 15 deg | 37.7 ± 14.8 | -7.8 | 0.75 | 15.6 | 0.20 |
| beacon_range | 3 | 42.4 ± 20.0 | -3.0 | 0.83 | 13.1 | 0.22 |
| beacon_range | 2 | 31.4 ± 18.7 | -14.0 | 0.75 | 14.3 | 0.18 |

largest drops: delay=3, beacon_range=2, dropout=0.15
