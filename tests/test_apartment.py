"""Stage 15: rectangular walls and apartment maps (Plan §20.2)."""

import math

import pytest

from doomworm.environments.maze import ApartmentConfig, apartment_world, rooms_visited
from doomworm.environments.simple_2d import AgentState, Wall, World
from doomworm.experiments.worm_agent import WormScenario


def test_wall_distance() -> None:
    w = Wall(2.0, 2.0, 4.0, 1.0)
    assert w.distance(3.0, 2.5) == 0.0
    assert w.distance(1.0, 2.5) == 1.0
    assert w.distance(7.0, 3.0) == 1.0
    assert w.distance(0.0, 0.0) == pytest.approx(math.hypot(2.0, 2.0))


def test_ray_hits_wall_and_collision_blocks() -> None:
    world = World(
        agent=AgentState(x=5.0, y=5.0), walls=[Wall(8.0, 0.0, 1.0, 20.0)], sensor_range=10
    )
    assert world.ray_distance(0.0) == pytest.approx(3.0)
    assert world.ray_distance(math.pi) == pytest.approx(5.0), "left wall of the box"
    assert world.ray_distance(math.pi / 2) == pytest.approx(10.0), "capped at sensor range"
    obs = world.observe()
    assert obs.sensor_front == pytest.approx(0.7)
    world.speed = 3.0
    world.step(1.0, 1.0)
    assert world.collisions == 1, "a 3-unit step would end inside the wall"
    assert world.agent.x == pytest.approx(5.0)
    world = World(agent=AgentState(x=7.4, y=5.0), walls=[Wall(8.0, 0.0, 1.0, 20.0)], speed=0.5)
    obs = world.step(1.0, 1.0)
    assert obs.collided
    assert world.agent.x == pytest.approx(7.4)


def test_ray_parallel_to_wall_misses() -> None:
    world = World(
        agent=AgentState(x=5.0, y=5.0), walls=[Wall(0.0, 8.0, 20.0, 1.0)], sensor_range=10
    )
    assert world.ray_distance(0.0) == pytest.approx(10.0), "sensor range: wall is above the ray"


@pytest.mark.parametrize("seed", range(8))
def test_apartment_layout_is_valid(seed: int) -> None:
    cfg = ApartmentConfig(n_food=2, furniture_per_room=1)
    w = apartment_world(seed, cfg)
    assert len(w.rooms) == 4
    assert not w._collides(w.agent)
    assert w.clearance(w.agent.x, w.agent.y) > cfg.margin - 1e-9
    for f in w.foods:
        assert w.clearance(f.x, f.y) > cfg.margin - 1e-9
    # every pair of adjacent rooms is joined by a door: a ray along the door centre crosses no wall
    partitions = [wl for wl in w.walls if wl.w == cfg.wall_thickness or wl.h == cfg.wall_thickness]
    assert len(partitions) == 8, "2x2 rooms: 4 partition segments split by 4 doors"
    assert apartment_world(seed, cfg).walls == w.walls


def test_doors_connect_rooms() -> None:
    cfg = ApartmentConfig(furniture_per_room=0)
    w = apartment_world(3, cfg)
    x_mid = cfg.width / 2
    partition = [wl for wl in w.walls if wl.w == cfg.wall_thickness and wl.x < x_mid < wl.x1]
    vertical = sorted(partition, key=lambda wl: wl.y)
    lower = [wl for wl in vertical if wl.y < cfg.height / 2]
    gap_y = (lower[0].y1 + lower[1].y) / 2
    world = World(agent=AgentState(x=x_mid - 3.0, y=gap_y), walls=w.walls, sensor_range=8)
    assert world.ray_distance(0.0) >= 6.0, "ray through the door centre reaches the next room"


def test_rooms_visited_metric() -> None:
    w = apartment_world(1, ApartmentConfig(furniture_per_room=0))
    assert rooms_visited(w, [(2.0, 2.0), (3.0, 3.0)]) == 1
    assert rooms_visited(w, [(2.0, 2.0), (15.0, 2.0), (15.0, 15.0)]) == 3
    assert w.room_index(50.0, 50.0) is None


def test_scenario_apartment_maps() -> None:
    sc = WormScenario(maps="apartment")
    a, b = sc.make_world(1), sc.make_world(2)
    assert a.walls != b.walls
    assert len(a.rooms) == 4
    assert len(a.foods) == 2
    assert sc.params["maps"] == "apartment"
    target = WormScenario(maps="apartment", task="target", dangers=1).make_world(1)
    assert target.target is not None
    assert len(target.dangers) == 1
