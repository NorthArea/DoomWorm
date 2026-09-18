# squeeze_genome

Aggregated from `runs/squeeze/benchmark` on the day `runs/` was cleared. Every number is the mean over the episodes of a row, then over seeds; the spread is across seeds. The brains that produced these rows are in `docs/doom/brains/`.

### Evaluated on `doom4`

| brain | seeds | reward | exits | kills | hits | shots | survived |
|---|---|---|---|---|---|---|---|
| interface | 3 | -0.63 ± 7.7 | 0.06 | 0.19 | 0.64 | 8.4 | 1.00 |
| all | 3 | -2.80 ± 7.9 | 0.00 | 0.17 | 0.56 | 20.3 | 0.92 |
| sensory | 3 | -13.01 ± 10.3 | 0.00 | 0.06 | 0.39 | 10.0 | 0.81 |

### Evaluated on `doom6`

| brain | seeds | reward | exits | kills | hits | shots | survived |
|---|---|---|---|---|---|---|---|
| interface | 3 | -31.55 ± 11.7 | 0.00 | 0.17 | 0.69 | 8.2 | 0.42 |
| all | 3 | -38.43 ± 15.9 | 0.00 | 0.19 | 1.08 | 16.1 | 0.36 |
| sensory | 3 | -40.66 ± 9.0 | 0.00 | 0.08 | 0.72 | 9.9 | 0.39 |

### Evaluated on `e1m1`

| brain | seeds | reward | exits | kills | hits | shots | survived |
|---|---|---|---|---|---|---|---|
| all | 3 | 4.83 ± 2.8 | 0.00 | 0.00 | 0.00 | 19.7 | 1.00 |
| sensory | 3 | 4.60 ± 2.1 | 0.00 | 0.00 | 0.00 | 8.0 | 1.00 |
| interface | 3 | 3.00 ± 0.4 | 0.00 | 0.00 | 0.00 | 26.7 | 1.00 |
