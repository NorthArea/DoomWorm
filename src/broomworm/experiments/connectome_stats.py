"""Stage 5 demo: load the Cook 2019 connectome and print what we got.

Run: ``uv run python -m broomworm.experiments.connectome_stats``
"""

from __future__ import annotations

from collections import Counter

import networkx as nx

from broomworm.connectome import GABA_NEURONS, ConnectionType, build_network, load_cook2019


def _top(degree: dict[str, int], k: int = 8) -> str:
    return ", ".join(f"{n}({d})" for n, d in sorted(degree.items(), key=lambda kv: -kv[1])[:k])


def main() -> int:
    """Print counts, hubs and a few sanity connections."""
    worm = load_cook2019()
    chem = worm.of_type(ConnectionType.CHEMICAL)
    gap = worm.of_type(ConnectionType.ELECTRICAL)

    print(f"neurons: {len(worm)}")
    for t, n in Counter(n.type for n in worm.neurons).most_common():
        print(f"  {t:<12}{n:>4}")
    print(f"chemical connections:   {len(chem):>5}  (weight sum {sum(c.weight for c in chem):.0f})")
    print(f"electrical connections: {len(gap):>5}  (directed; {len(gap) // 2} symmetric pairs)")

    g = nx.DiGraph()
    g.add_nodes_from(worm.ids())
    for c in chem:
        g.add_edge(c.source, c.target, weight=c.weight)
    largest_scc = max(map(len, nx.strongly_connected_components(g)))
    print(
        f"\nchemical graph: weakly connected = {nx.is_weakly_connected(g)}, "
        f"largest strongly connected component = {largest_scc}"
    )
    print("top in-degree:  " + _top(dict(g.in_degree())))
    print("top out-degree: " + _top(dict(g.out_degree())))

    print("\nknown connections:")
    lookup = {(c.source, c.target, c.connection_type): c.weight for c in worm.connections}
    for a, b in [("ASHL", "AVAL"), ("AWCL", "AIYL"), ("AVBL", "VB02"), ("AVAL", "VA02")]:
        chem_w = lookup.get((a, b, ConnectionType.CHEMICAL))
        gap_w = lookup.get((a, b, ConnectionType.ELECTRICAL))
        print(f"  {a} -> {b}: chemical {chem_w}, electrical {gap_w}")

    net = build_network(worm)
    negative = sum(1 for s in net.synapses if s.weight < 0)
    gaba_present = sum(1 for n in worm.ids() if n in GABA_NEURONS)
    print(
        f"\nsimulated network: {len(net.neurons)} neurons, {len(net.synapses)} synapses, "
        f"{negative} inhibitory synapses from {gaba_present} GABA neurons"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
