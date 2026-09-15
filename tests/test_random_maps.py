"""Stage 12: seeded random maps (Plan §18)."""

import math

import pytest

from doomworm.environments.maze import MapConfig, random_world
from doomworm.experiments.worm_agent import WormScenario


def test_random_world_is_deterministic_and_seed_dependent() -> None:
    a, b, c = random_world(7), random_world(7), random_world(8)
    assert [(o.x, o.y, o.radius) for o in a.obstacles] == [
        (o.x, o.y, o.radius) for o in b.obstacles
    ]
    assert a.agent == b.agent
    assert [(f.x, f.y) for f in a.foods] == [(f.x, f.y) for f in b.foods]
    assert a.obstacles != c.obstacles


@pytest.mark.parametrize("seed", range(10))
def test_layout_is_valid(seed: int) -> None:
    cfg = MapConfig(n_obstacles=6, n_food=3)
    w = random_world(seed, cfg)
    assert len(w.obstacles) == 6
    assert len(w.foods) == 3
    for i, o in enumerate(w.obstacles):
        assert o.radius <= o.x <= w.width - o.radius
        assert o.radius <= o.y <= w.height - o.radius
        for other in w.obstacles[i + 1 :]:
            assert math.dist((o.x, o.y), (other.x, other.y)) > o.radius + other.radius
        assert math.dist((w.agent.x, w.agent.y), (o.x, o.y)) > o.radius + w.agent_radius
    assert not w._collides(w.agent)
    for f in w.foods:
        assert all(math.dist((f.x, f.y), (o.x, o.y)) > o.radius for o in w.obstacles)


def test_respawn_uses_the_map_seed() -> None:
    w = random_world(3)
    first = [(f.x, f.y) for f in w.foods]
    w.foods = []
    w.foods = [w.spawn_food()]
    again = random_world(3)
    again.foods = []
    again.foods = [again.spawn_food()]
    assert first != [(f.x, f.y) for f in w.foods]
    assert [(f.x, f.y) for f in w.foods] == [(f.x, f.y) for f in again.foods]


def test_worm_scenario_random_maps() -> None:
    sc = WormScenario(maps="random")
    a, b = sc.make_world(1), sc.make_world(2)
    assert len(a.obstacles) == 5
    assert a.obstacles != b.obstacles
    assert sc.params["maps"] == "random"
    with pytest.raises(ValueError, match="maps"):
        WormScenario(maps="nope")
