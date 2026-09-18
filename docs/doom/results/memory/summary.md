# memory

Aggregated from `runs/memory/benchmark` on the day `runs/` was cleared. Every number is the mean over the episodes of a row, then over seeds; the spread is across seeds. The brains that produced these rows are in `docs/doom/brains/`.

### Evaluated on `doom1`

| brain | seeds | reward | exits | kills | hits | shots | survived |
|---|---|---|---|---|---|---|---|
| worm | 3 | 4.57 ± 0.9 | 0.03 | 0.00 | 0.00 | 0.0 | 1.00 |
| worm_memory+memory | 3 | 2.93 ± 2.7 | 0.03 | 0.00 | 0.00 | 0.0 | 1.00 |
| worm_shuffled_memory+memory | 3 | 2.06 ± 2.7 | 0.06 | 0.00 | 0.00 | 0.0 | 1.00 |
| worm_shuffled | 3 | 1.49 ± 0.6 | 0.00 | 0.00 | 0.00 | 0.0 | 1.00 |

### Evaluated on `doom2`

| brain | seeds | reward | exits | kills | hits | shots | survived |
|---|---|---|---|---|---|---|---|
| worm_memory+memory | 3 | 1.28 ± 0.6 | 0.00 | 0.00 | 0.00 | 0.0 | 1.00 |
| worm | 3 | 0.79 ± 1.5 | 0.00 | 0.00 | 0.00 | 0.0 | 1.00 |
| worm_shuffled | 3 | -0.39 ± 2.6 | 0.00 | 0.00 | 0.00 | 0.0 | 1.00 |
| worm_shuffled_memory+memory | 3 | -1.54 ± 3.0 | 0.00 | 0.00 | 0.00 | 0.0 | 1.00 |

### Evaluated on `doom3`

| brain | seeds | reward | exits | kills | hits | shots | survived |
|---|---|---|---|---|---|---|---|
| worm_shuffled | 3 | -6.70 ± 6.0 | 0.00 | 0.00 | 0.00 | 0.0 | 0.92 |
| worm | 3 | -18.70 ± 4.7 | 0.00 | 0.00 | 0.00 | 0.0 | 0.72 |
| worm_shuffled_memory+memory | 3 | -19.08 ± 6.0 | 0.00 | 0.00 | 0.00 | 0.0 | 0.78 |
| worm_memory+memory | 3 | -22.19 ± 7.5 | 0.00 | 0.00 | 0.00 | 0.0 | 0.61 |

### Evaluated on `doom4`

| brain | seeds | reward | exits | kills | hits | shots | survived |
|---|---|---|---|---|---|---|---|
| worm_shuffled | 3 | -2.15 ± 5.6 | 0.00 | 0.06 | 0.25 | 45.9 | 0.94 |
| worm | 3 | -2.80 ± 7.9 | 0.00 | 0.17 | 0.56 | 20.3 | 0.92 |
| worm_shuffled_memory+memory | 3 | -6.86 ± 6.6 | 0.03 | 0.00 | 0.03 | 49.2 | 0.92 |
| worm_memory+memory | 3 | -8.36 ± 1.8 | 0.00 | 0.08 | 0.56 | 2.5 | 0.83 |

### Evaluated on `doom5`

| brain | seeds | reward | exits | kills | hits | shots | survived |
|---|---|---|---|---|---|---|---|
| worm_memory+memory | 3 | -23.92 ± 5.6 | 0.00 | 0.14 | 0.69 | 3.5 | 0.56 |
| worm_shuffled | 3 | -25.67 ± 2.2 | 0.00 | 0.08 | 0.53 | 35.2 | 0.56 |
| worm | 3 | -28.81 ± 6.6 | 0.00 | 0.03 | 0.64 | 15.1 | 0.50 |
| worm_shuffled_memory+memory | 3 | -33.54 ± 10.5 | 0.00 | 0.06 | 0.31 | 37.9 | 0.47 |

### Evaluated on `doom6`

| brain | seeds | reward | exits | kills | hits | shots | survived |
|---|---|---|---|---|---|---|---|
| worm_memory+memory | 3 | -24.28 ± 7.3 | 0.00 | 0.39 | 1.69 | 9.9 | 0.50 |
| worm_shuffled | 3 | -28.12 ± 8.4 | 0.00 | 0.03 | 0.47 | 37.0 | 0.53 |
| worm_shuffled_memory+memory | 3 | -35.17 ± 4.9 | 0.00 | 0.03 | 0.14 | 38.4 | 0.50 |
| worm | 3 | -38.43 ± 15.9 | 0.00 | 0.19 | 1.08 | 16.1 | 0.36 |
