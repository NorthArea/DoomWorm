"""Stage 10: evolve the connectome weights in place (Plan §16)."""

from pathlib import Path

import numpy as np
import pytest

from doomworm.brain import load_brain
from doomworm.connectome import load_cook2019
from doomworm.experiments.evolve_worm import train_worm
from doomworm.experiments.worm_agent import WormScenario
from doomworm.learning import EvolutionConfig, evolve, seeded_population


def test_seeded_population_keeps_the_original_first() -> None:
    rng = np.random.default_rng(0)
    initial = np.array([0.1, -0.2, 0.3])
    pop = seeded_population(initial, size=5, sigma=0.01, rng=rng)
    assert pop.shape == (5, 3)
    assert pop[0] == pytest.approx(initial)
    assert np.abs(pop[1:] - initial).max() < 0.05


def test_evolve_from_initial_genome() -> None:
    target = np.array([0.5, -0.5])
    result = evolve(
        lambda w: -float(np.sum((w - target) ** 2)),
        n_weights=2,
        config=EvolutionConfig(population=12, generations=15, mutation_sigma=0.05),
        seed=1,
        initial=np.array([0.3, -0.3]),
    )
    assert result.best_weights == pytest.approx(target, abs=0.1)
    with pytest.raises(ValueError, match="initial genome"):
        evolve(lambda w: 0.0, n_weights=3, initial=np.zeros(2))


def test_train_worm_small_budget(tmp_path: Path) -> None:
    scenario = WormScenario(connectome=load_cook2019())
    initial = np.array(scenario.template.get_weights())
    cfg = EvolutionConfig(population=6, generations=3, mutation_sigma=0.02, weight_range=(-1, 1))
    out = tmp_path / "worm.json"
    result = train_worm(
        scenario,
        cfg,
        train_seeds=(1001,),
        steps=60,
        seed=0,
        out=out,
        metrics=out.with_suffix(".csv"),
    )
    assert len(result.history) == 3
    assert result.history[-1].best >= result.history[0].best
    net, meta = load_brain(out)
    assert meta["scenario"] == "worm"
    assert meta["params"]["brain_steps"] == 5
    assert len(net.get_weights()) == initial.size
    assert (
        not np.allclose(net.get_weights(), initial) or result.best_fitness == result.history[0].best
    )
    assert out.with_suffix(".csv").read_text().startswith("generation,best,mean")
