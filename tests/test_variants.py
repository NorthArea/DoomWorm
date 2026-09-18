"""Stage 11: control topologies and the comparison harness (Plan §17, §38, §49)."""

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from wormlab.connectome import (
    ConnectionType,
    Connectome,
    dense_connectome,
    load_cook2019,
    make_variant,
    random_connectome,
    shuffled_connectome,
)
from wormlab.experiments.compare_topologies import compare, summary_table
from wormlab.learning import EvolutionConfig, evolve


@pytest.fixture(scope="module")
def worm() -> Connectome:
    return load_cook2019()


def counts(c: Connectome) -> tuple[int, int]:
    return len(c.of_type(ConnectionType.CHEMICAL)), len(c.of_type(ConnectionType.ELECTRICAL))


def weight_multiset(c: Connectome, kind: ConnectionType) -> Counter[float]:
    return Counter(x.weight for x in c.of_type(kind))


def test_random_keeps_size_and_weights(worm: Connectome) -> None:
    r = random_connectome(worm, seed=1)
    assert r.ids() == worm.ids()
    assert counts(r) == counts(worm)
    assert weight_multiset(r, ConnectionType.CHEMICAL) == weight_multiset(
        worm, ConnectionType.CHEMICAL
    )
    real_edges = {(c.source, c.target) for c in worm.of_type(ConnectionType.CHEMICAL)}
    rand_edges = {(c.source, c.target) for c in r.of_type(ConnectionType.CHEMICAL)}
    assert len(real_edges & rand_edges) < 0.1 * len(real_edges)
    gap = {(c.source, c.target): c.weight for c in r.of_type(ConnectionType.ELECTRICAL)}
    assert all(gap.get((b, a)) == w for (a, b), w in gap.items())
    assert random_connectome(worm, seed=1).connections == r.connections


def test_shuffled_preserves_degrees(worm: Connectome) -> None:
    s = shuffled_connectome(worm, seed=2)
    assert counts(s) == counts(worm)
    for kind in (ConnectionType.CHEMICAL, ConnectionType.ELECTRICAL):
        assert weight_multiset(s, kind) == weight_multiset(worm, kind)
        for nid in worm.ids():
            out_real = sum(1 for c in worm.outgoing(nid) if c.connection_type is kind)
            out_shuf = sum(1 for c in s.outgoing(nid) if c.connection_type is kind)
            in_real = sum(1 for c in worm.incoming(nid) if c.connection_type is kind)
            in_shuf = sum(1 for c in s.incoming(nid) if c.connection_type is kind)
            assert (out_real, in_real) == (out_shuf, in_shuf), (nid, kind)
    real_edges = {(c.source, c.target) for c in worm.of_type(ConnectionType.CHEMICAL)}
    shuf_edges = {(c.source, c.target) for c in s.of_type(ConnectionType.CHEMICAL)}
    assert len(real_edges & shuf_edges) < 0.3 * len(real_edges), "most edges moved"


def test_dense_has_all_pairs_and_real_count_nonzero(worm: Connectome) -> None:
    d = dense_connectome(worm, seed=3)
    n = len(worm)
    chem = d.of_type(ConnectionType.CHEMICAL)
    assert len(chem) == n * n
    assert sum(1 for c in chem if c.weight) == len(worm.of_type(ConnectionType.CHEMICAL))
    assert not d.of_type(ConnectionType.ELECTRICAL)


def test_make_variant_dispatch(worm: Connectome) -> None:
    assert make_variant(worm, "real", 0) is worm
    with pytest.raises(ValueError, match="unknown variant"):
        make_variant(worm, "nope", 0)


def test_mutation_fraction_masks_genes() -> None:
    touched: list[np.ndarray] = []

    def fitness(w: np.ndarray) -> float:
        touched.append(w.copy())
        return 0.0

    cfg = EvolutionConfig(population=4, generations=2, mutation_sigma=1.0, mutation_fraction=0.1)
    evolve(fitness, n_weights=1000, config=cfg, seed=0, initial=np.zeros(1000))
    children = [w for w in touched[4:] if np.any(w != 0.0)]
    assert children
    changed = np.mean([np.mean(w != 0.0) for w in children])
    assert 0.03 < changed < 0.25


def test_compare_tiny_budget(tmp_path: Path) -> None:
    cfg = EvolutionConfig(population=4, generations=2, mutation_sigma=0.02, weight_range=(-1, 1))
    results = compare(
        ["real", "random"], cfg, train_seeds=(1001,), test_seeds=(1002,), steps=40, seed=0,
        out_dir=tmp_path,
    )  # fmt: skip
    assert [r.variant for r in results] == ["real", "random"]
    assert all(len(r.best_curve) == 2 for r in results)
    assert (tmp_path / "summary.md").read_text().startswith("| variant |")
    assert (tmp_path / "curves.png").stat().st_size > 1000
    assert (tmp_path / "real.json").exists()
    assert (tmp_path / "random_result.json").exists()
    assert "| real |" in summary_table(results)
