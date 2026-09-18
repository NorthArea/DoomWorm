# squeeze_search

Aggregated from `runs/squeeze2/benchmark` on the day `runs/` was cleared. Every number is the mean over the episodes of a row, then over seeds; the spread is across seeds. The brains that produced these rows are in `docs/doom/brains/`.

### Evaluated on `doom4`

| brain | seeds | reward | exits | kills | hits | shots | survived |
|---|---|---|---|---|---|---|---|
| cma_interface | 3 | 0.66 ± 3.4 | 0.00 | 0.08 | 0.31 | 36.2 | 0.97 |
| cma | 3 | -0.56 ± 1.1 | 0.00 | 0.03 | 0.14 | 50.0 | 1.00 |
| cma_interface_gains | 3 | -6.93 ± 7.3 | 0.00 | 0.00 | 0.08 | 32.8 | 0.86 |

### Evaluated on `doom6`

| brain | seeds | reward | exits | kills | hits | shots | survived |
|---|---|---|---|---|---|---|---|
| cma | 3 | -22.10 ± 3.5 | 0.00 | 0.11 | 0.53 | 39.2 | 0.64 |
| cma_interface_gains | 3 | -26.15 ± 7.0 | 0.00 | 0.11 | 0.67 | 23.6 | 0.56 |
| cma_interface | 3 | -34.48 ± 9.0 | 0.00 | 0.06 | 0.56 | 22.8 | 0.39 |

### Evaluated on `e1m1`

| brain | seeds | reward | exits | kills | hits | shots | survived |
|---|---|---|---|---|---|---|---|
| cma_interface_gains | 3 | 2.33 ± 1.2 | 0.00 | 0.00 | 0.00 | 50.0 | 1.00 |
| cma_interface | 3 | 1.10 ± 0.7 | 0.00 | 0.00 | 0.00 | 48.7 | 1.00 |
| cma | 3 | 0.63 ± 0.3 | 0.00 | 0.00 | 0.00 | 50.0 | 1.00 |
