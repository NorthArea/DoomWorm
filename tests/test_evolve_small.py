"""Stage 4 acceptance: evolved weights reach food more often than random ones (Plan §10)."""

from pathlib import Path

from doomworm.brain import load_brain
from doomworm.experiments.evolve_small import SmallFoodScenario, train
from doomworm.learning import EvolutionConfig, evaluate

CFG = EvolutionConfig(population=16, generations=8, elite_fraction=0.25, mutation_sigma=0.3)


def test_scenario_worlds_differ_by_seed() -> None:
    sc = SmallFoodScenario()
    a, b = sc.make_world(1), sc.make_world(2)
    assert (a.agent, a.foods) != (b.agent, b.foods)
    assert sc.make_world(1).agent == a.agent


def test_training_improves_fitness_and_saves_brain(tmp_path: Path) -> None:
    out = tmp_path / "brain.json"
    result = train(CFG, train_seeds=(0, 1), steps=300, seed=0, out=out)
    assert result.history[-1].best > result.history[0].mean
    net, meta = load_brain(out)
    assert meta["stage"] == 4
    assert meta["scenario"] == "small_food"
    assert net.get_weights() == list(result.best_weights)


def test_evolved_brain_generalises_to_unseen_seed(tmp_path: Path) -> None:
    result = train(CFG, train_seeds=(0, 1), steps=300, seed=0, out=tmp_path / "b.json")
    sc = SmallFoodScenario()
    evolved = evaluate(sc, result.best_weights, seeds=(100, 101), steps=300)
    random_scores = [
        evaluate(sc, sc.random_weights(seed=s), seeds=(100, 101), steps=300) for s in range(4)
    ]
    assert evolved.fitness > max(r.fitness for r in random_scores)
    assert evolved.food_eaten >= 1
