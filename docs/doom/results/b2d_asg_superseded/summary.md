# b2d_asg_superseded

Aggregated from `runs/benchmark_doom_b2d_asg` on the day `runs/` was cleared. Every number is the mean over the episodes of a row, then over seeds; the spread is across seeds. The brains that produced these rows are in `docs/doom/brains/`.

### Evaluated on `doom1`

| brain | seeds | reward | exits | kills | hits | shots | survived |
|---|---|---|---|---|---|---|---|
| worm | 5 | 5.24 ± 3.9 | 0.22 | 0.00 | 0.00 | 0.0 | 1.00 |
| worm_shuffled | 4 | 5.13 ± 5.2 | 0.19 | 0.00 | 0.00 | 0.0 | 1.00 |
| worm_from_worm_evolved_random | 5 | 4.86 ± 4.3 | 0.17 | 0.00 | 0.00 | 0.0 | 1.00 |
| worm_random | 4 | 1.16 ± 0.4 | 0.00 | 0.00 | 0.00 | 0.0 | 1.00 |

### Evaluated on `doom2`

| brain | seeds | reward | exits | kills | hits | shots | survived |
|---|---|---|---|---|---|---|---|
| worm_from_worm_evolved_random | 5 | 2.75 ± 1.8 | 0.08 | 0.00 | 0.00 | 0.0 | 1.00 |
| worm | 5 | 1.66 ± 0.9 | 0.05 | 0.00 | 0.00 | 0.0 | 1.00 |
| worm_shuffled | 4 | 1.37 ± 0.8 | 0.00 | 0.00 | 0.00 | 0.0 | 1.00 |
| worm_random | 4 | -0.70 ± 1.9 | 0.00 | 0.00 | 0.00 | 0.0 | 1.00 |

### Evaluated on `doom3`

| brain | seeds | reward | exits | kills | hits | shots | survived |
|---|---|---|---|---|---|---|---|
| worm_shuffled | 4 | -13.38 ± 6.6 | 0.00 | 0.00 | 0.00 | 0.0 | 0.81 |
| worm_random | 4 | -13.63 ± 9.1 | 0.00 | 0.00 | 0.00 | 0.0 | 0.81 |
| worm | 5 | -17.53 ± 10.3 | 0.02 | 0.00 | 0.00 | 0.0 | 0.78 |
| worm_from_worm_evolved_random | 5 | -23.38 ± 2.9 | 0.03 | 0.00 | 0.00 | 0.0 | 0.60 |

### Evaluated on `doom4`

| brain | seeds | reward | exits | kills | hits | shots | survived |
|---|---|---|---|---|---|---|---|
| worm_random | 4 | -0.99 ± 3.5 | 0.00 | 0.10 | 0.35 | 33.7 | 1.00 |
| worm_from_worm_evolved_random | 5 | -1.63 ± 7.1 | 0.03 | 0.20 | 0.73 | 18.8 | 0.92 |
| worm | 5 | -6.46 ± 8.5 | 0.00 | 0.07 | 0.30 | 8.0 | 0.92 |
| worm_shuffled | 4 | -8.07 ± 10.6 | 0.00 | 0.04 | 0.21 | 48.8 | 0.88 |

### Evaluated on `doom5`

| brain | seeds | reward | exits | kills | hits | shots | survived |
|---|---|---|---|---|---|---|---|
| worm_from_worm_evolved_random | 5 | -20.93 ± 9.0 | 0.02 | 0.17 | 0.90 | 12.7 | 0.60 |
| worm_random | 4 | -28.92 ± 3.9 | 0.00 | 0.08 | 0.73 | 25.3 | 0.48 |
| worm_shuffled | 4 | -29.13 ± 5.2 | 0.00 | 0.02 | 0.56 | 34.3 | 0.50 |
| worm | 5 | -30.28 ± 9.3 | 0.00 | 0.07 | 0.57 | 7.0 | 0.50 |

### Evaluated on `doom6`

| brain | seeds | reward | exits | kills | hits | shots | survived |
|---|---|---|---|---|---|---|---|
| worm_random | 4 | -22.65 ± 7.3 | 0.00 | 0.12 | 0.60 | 35.9 | 0.62 |
| worm_from_worm_evolved_random | 5 | -31.44 ± 9.4 | 0.02 | 0.32 | 1.50 | 15.9 | 0.45 |
| worm | 5 | -33.89 ± 10.1 | 0.00 | 0.17 | 0.88 | 17.3 | 0.50 |
| worm_shuffled | 4 | -39.93 ± 12.3 | 0.00 | 0.00 | 0.46 | 34.7 | 0.38 |
