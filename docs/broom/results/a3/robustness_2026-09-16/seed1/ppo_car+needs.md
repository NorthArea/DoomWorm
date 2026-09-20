# ppo_car+needs around the car preset

| parameter | value | reward | vs base | survived | collisions | coverage |
|---|---|---|---|---|---|---|
| base | (preset) | 56.9 ± 11.6 | +0.0 | 0.92 | 7.9 | 0.28 |
| noise_sigma | 0.15 | 55.0 ± 20.5 | -1.9 | 1.00 | 33.1 | 0.27 |
| noise_sigma | 0.3 | 35.2 ± 13.8 | -21.7 | 0.58 | 21.6 | 0.22 |
| dropout | 0.15 | 46.2 ± 20.4 | -10.7 | 0.75 | 22.1 | 0.27 |
| dropout | 0.3 | 33.4 ± 17.3 | -23.6 | 0.67 | 63.8 | 0.24 |
| odom_sigma | 0.2 | 47.0 ± 19.4 | -9.9 | 0.67 | 28.7 | 0.27 |
| odom_sigma | 0.4 | 41.7 ± 19.5 | -15.3 | 0.67 | 26.5 | 0.25 |
| delay | 2 | 44.4 ± 20.4 | -12.5 | 0.67 | 32.9 | 0.25 |
| delay | 3 | 25.0 ± 17.6 | -31.9 | 0.42 | 17.9 | 0.20 |
| beacon_fov | 20 deg | 60.7 ± 21.4 | +3.8 | 0.92 | 29.2 | 0.29 |
| beacon_fov | 15 deg | 50.5 ± 21.5 | -6.5 | 0.75 | 11.0 | 0.27 |
| beacon_range | 3 | 42.9 ± 21.8 | -14.0 | 0.67 | 16.0 | 0.24 |
| beacon_range | 2 | 40.0 ± 23.8 | -16.9 | 0.50 | 16.8 | 0.23 |

largest drops: delay=3, dropout=0.3, noise_sigma=0.3
