# Cook et al. 2019 hermaphrodite connectome

Source: WormWiring, Emmons lab, https://wormwiring.org/pages/adjacency.html

- `SI 5 Connectome adjacency matrices, corrected July 2020.xlsx` (sheets
  `hermaphrodite chemical`, `hermaphrodite gap jn symmetric`)
- `SI 4 Cell lists.xlsx` (cell types)

Citation: Cook, S.J., Jarrell, T.A., Brittin, C.A. et al. Whole-animal
connectomes of both Caenorhabditis elegans sexes. Nature 571, 63–71 (2019).
https://doi.org/10.1038/s41586-019-1352-7

Retrieved on the date in `RETRIEVED` by `scripts/fetch_connectome.py`
(`uv run --with openpyxl scripts/fetch_connectome.py`).

License: the WormWiring site states no explicit license; the tables are the
Supplementary Information of the article above and are used here for
non-commercial research. Cite the paper when publishing results.

## Files

| file | rows | meaning |
|------|------|---------|
| `neurons.csv` | 302 | `name, type, category, notes`; type is `sensory`, `interneuron`, `motorneuron` or `neuron` (CANL/CANR, unclassified) |
| `chemical.csv` | 3709 | directed `source -> target` chemical synapses; `weight` = number of EM serial sections of connectivity (synapse count x size), not a synapse count |
| `gap_junction.csv` | 1105 | undirected pairs `a, b` with `a <= b`; 14 rows have `a == b` (self entries present in the source table) |

Only neuron-to-neuron entries are kept. Muscles, glia, pharyngeal epithelium
and other end organs are dropped. 38 chemical self-loops are kept as in the
source. Weights are integers 1..75 (median 3).
