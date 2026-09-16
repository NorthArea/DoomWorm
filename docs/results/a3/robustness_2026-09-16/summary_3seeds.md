
### worm_from_worm_evolved_random+needs (3 seeds)

| parameter | value | reward mean ± std over seeds | vs base | survived | collisions |
|---|---|---|---|---|---|
| base | (preset) | 41.3 ± 3.2 | +0.0 | 0.92 | 14.9 |
| noise_sigma | 0.15 | 39.5 ± 4.8 | -1.8 | 0.86 | 13.7 |
| noise_sigma | 0.3 | 39.0 ± 2.4 | -2.2 | 0.92 | 9.9 |
| dropout | 0.15 | 36.0 ± 2.7 | -5.3 | 0.89 | 23.7 |
| dropout | 0.3 | 36.5 ± 2.2 | -4.8 | 0.92 | 30.3 |
| odom_sigma | 0.2 | 36.1 ± 1.9 | -5.2 | 0.75 | 16.8 |
| odom_sigma | 0.4 | 29.7 ± 5.1 | -11.6 | 0.69 | 14.7 |
| delay | 2 | 37.4 ± 5.2 | -3.9 | 0.89 | 17.6 |
| delay | 3 | 30.0 ± 5.1 | -11.3 | 0.81 | 21.5 |
| beacon_fov | 20 deg | 39.9 ± 5.5 | -1.4 | 0.86 | 15.6 |
| beacon_fov | 15 deg | 36.2 ± 4.7 | -5.0 | 0.78 | 18.3 |
| beacon_range | 3 | 40.0 ± 2.7 | -1.3 | 0.80 | 14.7 |
| beacon_range | 2 | 28.9 ± 2.1 | -12.3 | 0.61 | 12.7 |

### ppo_car+needs (3 seeds)

| parameter | value | reward mean ± std over seeds | vs base | survived | collisions |
|---|---|---|---|---|---|
| base | (preset) | 47.8 ± 6.5 | +0.0 | 0.92 | 23.1 |
| noise_sigma | 0.15 | 45.0 ± 8.6 | -2.8 | 0.92 | 33.9 |
| noise_sigma | 0.3 | 37.2 ± 2.2 | -10.5 | 0.69 | 24.9 |
| dropout | 0.15 | 45.8 ± 2.9 | -1.9 | 0.83 | 27.9 |
| dropout | 0.3 | 41.4 ± 5.9 | -6.3 | 0.78 | 42.9 |
| odom_sigma | 0.2 | 46.1 ± 0.7 | -1.7 | 0.81 | 28.0 |
| odom_sigma | 0.4 | 42.0 ± 2.1 | -5.7 | 0.67 | 24.0 |
| delay | 2 | 40.7 ± 2.6 | -7.0 | 0.78 | 25.0 |
| delay | 3 | 29.0 ± 3.3 | -18.8 | 0.50 | 17.9 |
| beacon_fov | 20 deg | 50.3 ± 8.3 | +2.6 | 0.92 | 28.1 |
| beacon_fov | 15 deg | 48.6 ± 3.6 | +0.8 | 0.86 | 16.9 |
| beacon_range | 3 | 42.3 ± 1.8 | -5.4 | 0.78 | 19.5 |
| beacon_range | 2 | 36.1 ± 3.0 | -11.7 | 0.61 | 22.6 |
