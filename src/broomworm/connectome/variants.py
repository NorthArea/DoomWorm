"""Control topologies for the comparison of Plan §17 / §49.

    real      the loaded connectome
    random    same neurons, same number of chemical and electrical connections,
              endpoints drawn uniformly, weights = a permutation of the real ones
    shuffled  degree-preserving edge swaps (in/out degree of every neuron kept for
              chemical, degree kept for electrical), weights permuted
    dense     every ordered pair is a chemical connection; the real number of
              them start with (permuted) real weights, the rest with 0. With a
              sparse mutation mask this is the Free mode of Plan §38: topology
              emerges from which weights become non-zero.

Self-loops present in the data are kept as they are in all variants.
"""

from __future__ import annotations

import numpy as np

from broomworm.connectome.model import Connection, ConnectionType, Connectome

VARIANTS = ("real", "random", "shuffled", "dense")


def _split(worm: Connectome) -> tuple[list[Connection], list[tuple[str, str, float]]]:
    """Chemical connections and undirected gap-junction pairs (a <= b)."""
    chem = worm.of_type(ConnectionType.CHEMICAL)
    pairs = {
        (min(c.source, c.target), max(c.source, c.target)): c.weight
        for c in worm.of_type(ConnectionType.ELECTRICAL)
    }
    return chem, [(a, b, w) for (a, b), w in sorted(pairs.items())]


def _electrical(pairs: list[tuple[str, str, float]]) -> list[Connection]:
    out = []
    for a, b, w in pairs:
        out.append(Connection(a, b, w, ConnectionType.ELECTRICAL))
        if a != b:
            out.append(Connection(b, a, w, ConnectionType.ELECTRICAL))
    return out


def random_connectome(worm: Connectome, seed: int) -> Connectome:
    """Same size, uniformly random endpoints, permuted weights (Plan §17 B)."""
    rng = np.random.default_rng(seed)
    ids = worm.ids()
    chem, pairs = _split(worm)

    chem_weights = rng.permutation([c.weight for c in chem])
    new_chem: list[Connection] = []
    seen: set[tuple[str, str]] = set()
    for w in chem_weights:
        while True:
            a, b = ids[rng.integers(len(ids))], ids[rng.integers(len(ids))]
            if (a, b) not in seen:
                seen.add((a, b))
                break
        new_chem.append(Connection(a, b, float(w), ConnectionType.CHEMICAL))

    # keep the number of self entries so the directed electrical count matches
    n_loops = sum(1 for a, b, _ in pairs if a == b)
    gap_weights = rng.permutation([w for _, _, w in pairs])
    new_pairs: list[tuple[str, str, float]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for k, w in enumerate(gap_weights):
        while True:
            if k < n_loops:
                a = b = ids[rng.integers(len(ids))]
            else:
                a, b = sorted((ids[rng.integers(len(ids))], ids[rng.integers(len(ids))]))
                if a == b:
                    continue
            if (a, b) not in seen_pairs:
                seen_pairs.add((a, b))
                break
        new_pairs.append((a, b, float(w)))
    return Connectome(neurons=list(worm.neurons), connections=new_chem + _electrical(new_pairs))


def _swap_directed(
    edges: list[tuple[str, str]], rng: np.random.Generator, rounds: int
) -> list[tuple[str, str]]:
    """Degree-preserving rewiring: (a->b, c->d) -> (a->d, c->b) when legal."""
    edges = list(edges)
    present = set(edges)
    n = len(edges)
    for _ in range(rounds * n):
        i, j = rng.integers(n), rng.integers(n)
        (a, b), (c, d) = edges[i], edges[j]
        if a == d or c == b or (a, d) in present or (c, b) in present:
            continue
        present -= {(a, b), (c, d)}
        present |= {(a, d), (c, b)}
        edges[i], edges[j] = (a, d), (c, b)
    return edges


def _swap_undirected(
    edges: list[tuple[str, str]], rng: np.random.Generator, rounds: int
) -> list[tuple[str, str]]:
    """Degree-preserving rewiring of an undirected simple graph."""
    edges = list(edges)
    present = {tuple(sorted(e)) for e in edges}
    n = len(edges)
    for _ in range(rounds * n):
        i, j = rng.integers(n), rng.integers(n)
        (a, b), (c, d) = edges[i], edges[j]
        if rng.integers(2):
            c, d = d, c
        if len({a, b, c, d}) < 4:
            continue
        e1, e2 = tuple(sorted((a, d))), tuple(sorted((c, b)))
        if e1 in present or e2 in present:
            continue
        present -= {tuple(sorted((a, b))), tuple(sorted((c, d)))}
        present |= {e1, e2}
        edges[i], edges[j] = (e1[0], e1[1]), (e2[0], e2[1])
    return edges


def shuffled_connectome(worm: Connectome, seed: int, rounds: int = 10) -> Connectome:
    """Degree-preserving shuffle, weights permuted (Plan §17 C)."""
    rng = np.random.default_rng(seed)
    chem, pairs = _split(worm)

    loops = [c for c in chem if c.source == c.target]
    plain = [(c.source, c.target) for c in chem if c.source != c.target]
    swapped = _swap_directed(plain, rng, rounds)
    weights = rng.permutation([c.weight for c in chem if c.source != c.target])
    new_chem = [
        Connection(a, b, float(w), ConnectionType.CHEMICAL)
        for (a, b), w in zip(swapped, weights, strict=True)
    ]
    new_chem += loops

    gap_loops = [(a, b, w) for a, b, w in pairs if a == b]
    gap_plain = [(a, b) for a, b, _ in pairs if a != b]
    gap_swapped = _swap_undirected(gap_plain, rng, rounds)
    gap_weights = rng.permutation([w for a, b, w in pairs if a != b])
    new_pairs = [(a, b, float(w)) for (a, b), w in zip(gap_swapped, gap_weights, strict=True)]
    new_pairs += gap_loops
    return Connectome(neurons=list(worm.neurons), connections=new_chem + _electrical(new_pairs))


def dense_connectome(worm: Connectome, seed: int) -> Connectome:
    """All ordered pairs as chemical connections; real count non-zero (Plan §38 Free)."""
    rng = np.random.default_rng(seed)
    ids = worm.ids()
    chem, _ = _split(worm)
    weights = np.zeros(len(ids) * len(ids))
    slots = rng.choice(weights.size, size=len(chem), replace=False)
    weights[slots] = rng.permutation([c.weight for c in chem])
    connections = [
        Connection(a, b, float(weights[i * len(ids) + j]), ConnectionType.CHEMICAL)
        for i, a in enumerate(ids)
        for j, b in enumerate(ids)
    ]
    return Connectome(neurons=list(worm.neurons), connections=connections)


def make_variant(worm: Connectome, name: str, seed: int) -> Connectome:
    """Dispatch by variant name."""
    if name == "real":
        return worm
    if name == "random":
        return random_connectome(worm, seed)
    if name == "shuffled":
        return shuffled_connectome(worm, seed)
    if name == "dense":
        return dense_connectome(worm, seed)
    raise ValueError(f"unknown variant {name!r}; choose from {VARIANTS}")
