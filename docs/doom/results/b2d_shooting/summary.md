# b2d_shooting

Aggregated from `runs/benchmark_doom_b2d` on the day `runs/` was cleared. Every number is the mean over the episodes of a row, then over seeds; the spread is across seeds. The brains that produced these rows are in `docs/doom/brains/`.

### Evaluated on `doom1`

| brain | seeds | reward | exits | kills | hits | shots | survived |
|---|---|---|---|---|---|---|---|
| worm_from_worm_evolved_random | 6 | 5.27 ± 6.0 | 0.26 | 0.00 | 0.00 | 0.0 | 1.00 |
| worm | 6 | 4.29 ± 1.8 | 0.10 | 0.00 | 0.00 | 0.0 | 1.00 |
| worm_random | 6 | 1.18 ± 8.4 | 0.21 | 0.00 | 0.00 | 0.0 | 1.00 |
| worm_shuffled | 6 | 0.10 ± 2.8 | 0.00 | 0.00 | 0.00 | 0.0 | 1.00 |

### Evaluated on `doom2`

| brain | seeds | reward | exits | kills | hits | shots | survived |
|---|---|---|---|---|---|---|---|
| worm_from_worm_evolved_random | 6 | 1.28 ± 2.9 | 0.11 | 0.00 | 0.00 | 0.0 | 1.00 |
| worm | 6 | 1.25 ± 1.1 | 0.00 | 0.00 | 0.00 | 0.0 | 1.00 |
| worm_shuffled | 6 | -0.64 ± 2.3 | 0.00 | 0.00 | 0.00 | 0.0 | 1.00 |
| worm_random | 6 | -2.21 ± 4.3 | 0.06 | 0.00 | 0.00 | 0.0 | 1.00 |

### Evaluated on `doom3`

| brain | seeds | reward | exits | kills | hits | shots | survived |
|---|---|---|---|---|---|---|---|
| worm_shuffled | 6 | -5.75 ± 5.1 | 0.00 | 0.00 | 0.00 | 0.0 | 0.94 |
| worm | 6 | -13.10 ± 7.2 | 0.00 | 0.00 | 0.00 | 0.0 | 0.85 |
| worm_random | 6 | -17.04 ± 7.0 | 0.03 | 0.00 | 0.00 | 0.0 | 0.78 |
| worm_from_worm_evolved_random | 6 | -21.84 ± 13.6 | 0.03 | 0.00 | 0.00 | 0.0 | 0.67 |

### Evaluated on `doom4`

| brain | seeds | reward | exits | kills | hits | shots | survived |
|---|---|---|---|---|---|---|---|
| worm_shuffled | 6 | -1.84 ± 3.9 | 0.00 | 0.03 | 0.18 | 47.8 | 0.97 |
| worm | 6 | -3.50 ± 5.7 | 0.00 | 0.08 | 0.31 | 19.6 | 0.93 |
| worm_random | 6 | -4.80 ± 4.4 | 0.03 | 0.04 | 0.24 | 29.0 | 0.96 |
| worm_from_worm_evolved_random | 6 | -7.04 ± 6.8 | 0.04 | 0.11 | 0.51 | 26.1 | 0.89 |

### Evaluated on `doom5`

| brain | seeds | reward | exits | kills | hits | shots | survived |
|---|---|---|---|---|---|---|---|
| worm_random | 6 | -27.61 ± 5.1 | 0.01 | 0.08 | 0.54 | 25.1 | 0.54 |
| worm | 6 | -28.28 ± 4.4 | 0.00 | 0.06 | 0.50 | 15.2 | 0.56 |
| worm_shuffled | 6 | -28.46 ± 3.5 | 0.00 | 0.04 | 0.33 | 34.6 | 0.53 |
| worm_from_worm_evolved_random | 6 | -36.41 ± 8.8 | 0.01 | 0.04 | 0.71 | 17.6 | 0.39 |

### Evaluated on `doom6`

| brain | seeds | reward | exits | kills | hits | shots | survived |
|---|---|---|---|---|---|---|---|
| worm_shuffled | 6 | -25.59 ± 9.7 | 0.00 | 0.04 | 0.43 | 37.7 | 0.60 |
| worm_random | 6 | -32.88 ± 8.8 | 0.01 | 0.04 | 0.54 | 26.9 | 0.49 |
| worm | 6 | -34.12 ± 11.5 | 0.00 | 0.11 | 0.76 | 17.7 | 0.47 |
| worm_from_worm_evolved_random | 6 | -35.65 ± 8.7 | 0.03 | 0.10 | 0.90 | 20.5 | 0.39 |

### Evaluated on `vizdoom1`

| brain | seeds | reward | exits | kills | hits | shots | survived |
|---|---|---|---|---|---|---|---|
| worm_from_worm_evolved_random | 6 | 5.26 ± 6.0 | 0.26 | 0.00 | 0.00 | 0.0 | 1.00 |
| worm | 6 | 4.31 ± 1.8 | 0.10 | 0.00 | 0.00 | 0.0 | 1.00 |
| worm_random | 6 | 1.19 ± 8.4 | 0.21 | 0.00 | 0.00 | 0.0 | 1.00 |
| worm_shuffled | 6 | 0.10 ± 2.8 | 0.00 | 0.00 | 0.00 | 0.0 | 1.00 |

### Evaluated on `vizdoom2`

| brain | seeds | reward | exits | kills | hits | shots | survived |
|---|---|---|---|---|---|---|---|
| worm_from_worm_evolved_random | 6 | 1.56 ± 2.8 | 0.10 | 0.00 | 0.00 | 0.0 | 1.00 |
| worm | 6 | 1.19 ± 1.3 | 0.00 | 0.00 | 0.00 | 0.0 | 1.00 |
| worm_shuffled | 6 | -0.62 ± 2.4 | 0.00 | 0.00 | 0.00 | 0.0 | 1.00 |
| worm_random | 6 | -1.93 ± 4.1 | 0.07 | 0.00 | 0.00 | 0.0 | 1.00 |

### Evaluated on `vizdoom3`

| brain | seeds | reward | exits | kills | hits | shots | survived |
|---|---|---|---|---|---|---|---|
| worm_shuffled | 6 | -5.54 ± 5.2 | 0.00 | 0.00 | 0.00 | 0.0 | 0.94 |
| worm | 6 | -13.21 ± 7.2 | 0.00 | 0.00 | 0.00 | 0.0 | 0.85 |
| worm_random | 6 | -17.93 ± 7.3 | 0.03 | 0.00 | 0.00 | 0.0 | 0.75 |
| worm_from_worm_evolved_random | 6 | -23.51 ± 12.7 | 0.03 | 0.00 | 0.00 | 0.0 | 0.62 |

### Evaluated on `vizdoom4`

| brain | seeds | reward | exits | kills | hits | shots | survived |
|---|---|---|---|---|---|---|---|
| worm_from_worm_evolved_random | 6 | -0.55 ± 5.0 | 0.07 | 0.21 | 0.21 | 26.1 | 0.99 |
| worm_shuffled | 6 | -1.44 ± 3.3 | 0.00 | 0.08 | 0.08 | 48.1 | 0.97 |
| worm | 6 | -2.26 ± 4.5 | 0.00 | 0.11 | 0.11 | 19.2 | 0.96 |
| worm_random | 6 | -3.52 ± 5.7 | 0.03 | 0.11 | 0.11 | 27.8 | 0.97 |

### Evaluated on `vizdoom5`

| brain | seeds | reward | exits | kills | hits | shots | survived |
|---|---|---|---|---|---|---|---|
| worm_from_worm_evolved_random | 6 | -2.81 ± 6.5 | 0.12 | 0.28 | 0.28 | 28.2 | 0.93 |
| worm_random | 6 | -2.90 ± 5.0 | 0.06 | 0.25 | 0.25 | 36.9 | 0.97 |
| worm_shuffled | 6 | -4.50 ± 4.9 | 0.00 | 0.15 | 0.15 | 49.3 | 0.94 |
| worm | 6 | -7.08 ± 11.4 | 0.00 | 0.12 | 0.12 | 20.6 | 0.92 |

### Evaluated on `vizdoom6`

| brain | seeds | reward | exits | kills | hits | shots | survived |
|---|---|---|---|---|---|---|---|
| worm_shuffled | 6 | -3.30 ± 4.2 | 0.00 | 0.18 | 0.18 | 49.5 | 0.99 |
| worm_from_worm_evolved_random | 6 | -4.69 ± 8.8 | 0.04 | 0.38 | 0.38 | 35.2 | 0.96 |
| worm | 6 | -5.39 ± 8.4 | 0.00 | 0.22 | 0.22 | 25.5 | 0.99 |
| worm_random | 6 | -6.28 ± 4.3 | 0.03 | 0.22 | 0.22 | 40.0 | 0.97 |
