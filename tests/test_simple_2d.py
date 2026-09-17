"""Stage 1: 2D world, differential drive, 3 distance sensors (Plan §7)."""

import math

import pytest

from doomworm.environments.simple_2d import AgentState, Food, Obstacle, World

DEG = math.pi / 180


def make_world(**kwargs: object) -> World:
    defaults: dict[str, object] = {
        "width": 20.0,
        "height": 20.0,
        "agent": AgentState(x=10.0, y=10.0, heading=0.0),
        "sensor_range": 4.0,
        "speed": 0.5,
    }
    defaults.update(kwargs)
    return World(**defaults)  # type: ignore[arg-type]


def test_forward_moves_along_heading() -> None:
    w = make_world()
    w.step(1.0, 1.0)
    assert w.agent.x == pytest.approx(10.5)
    assert w.agent.y == pytest.approx(10.0)
    assert w.agent.heading == pytest.approx(0.0)


def test_differential_drive_turns() -> None:
    left = make_world()
    left.step(0.0, 1.0)
    assert left.agent.heading > 0.0, "right motor only -> turn left (CCW)"

    right = make_world()
    right.step(1.0, 0.0)
    assert right.agent.heading < 0.0, "left motor only -> turn right (CW)"


def test_motors_are_clamped() -> None:
    w = make_world()
    w.step(5.0, 5.0)  # clamped to (1, 1): full speed ahead
    assert math.dist((w.agent.x, w.agent.y), (10.0, 10.0)) == pytest.approx(0.5)
    assert w.agent.heading == pytest.approx(0.0)


def test_no_obstacle_in_range_reads_zero() -> None:
    w = make_world()
    obs = w.observe()
    assert obs.sensor_left == obs.sensor_front == obs.sensor_right == 0.0


def test_front_sensor_sees_obstacle_ahead() -> None:
    w = make_world(obstacles=[Obstacle(x=13.0, y=10.0, radius=1.0)])
    obs = w.observe()
    assert obs.sensor_front == pytest.approx(0.5)  # surface at 2.0 of 4.0 range
    assert obs.sensor_left < obs.sensor_front
    assert obs.sensor_right < obs.sensor_front


def test_side_sensors_are_lateralised() -> None:
    # Obstacle up-left of an east-facing agent.
    w = make_world(obstacles=[Obstacle(x=12.0, y=12.0, radius=1.0)])
    obs = w.observe()
    assert obs.sensor_left > 0.0
    assert obs.sensor_right == 0.0


def test_walls_are_sensed() -> None:
    w = make_world(agent=AgentState(x=18.5, y=10.0, heading=0.0))
    assert w.observe().sensor_front > 0.0


def test_collision_blocks_movement_and_counts() -> None:
    w = make_world(obstacles=[Obstacle(x=11.0, y=10.0, radius=0.4)])
    obs = w.step(1.0, 1.0)
    assert obs.collided
    assert w.collisions == 1
    assert w.agent.x == pytest.approx(10.0)


def test_cannot_leave_world() -> None:
    w = make_world(agent=AgentState(x=19.4, y=10.0, heading=0.0))
    obs = w.step(1.0, 1.0)
    assert obs.collided
    assert w.agent.x == pytest.approx(19.4)


def test_observation_channels_are_floats() -> None:
    ch = make_world().observe().as_channels()
    assert all(isinstance(v, float) for v in ch.values())


# --- Stage 2: food + hunger (Plan §8) ------------------------------------------


def test_no_food_reads_zero() -> None:
    obs = make_world().observe()
    assert obs.food_left == obs.food_front == obs.food_right == 0.0


def test_food_ahead_reads_inverse_distance() -> None:
    w = make_world(foods=[Food(x=14.0, y=10.0)])
    obs = w.observe()
    assert obs.food_front == pytest.approx(0.25)
    assert obs.food_left == obs.food_right == 0.0


def test_food_signal_is_clamped_to_one() -> None:
    w = make_world(foods=[Food(x=10.5, y=10.0)])
    assert w.observe().food_front == 1.0


def test_food_sensors_are_sectored() -> None:
    left = make_world(foods=[Food(x=10.0, y=12.0)]).observe()  # bearing +90°
    assert (left.food_left > 0.0, left.food_front, left.food_right) == (True, 0.0, 0.0)

    right = make_world(foods=[Food(x=10.0, y=8.0)]).observe()  # bearing -90°
    assert (right.food_right > 0.0, right.food_front, right.food_left) == (True, 0.0, 0.0)

    behind = make_world(foods=[Food(x=8.0, y=10.0 + 1e-9)]).observe()  # bearing ~180°
    assert behind.food_left > 0.0, "food behind is still sensed on the left sector"


def test_only_nearest_food_is_sensed() -> None:
    w = make_world(foods=[Food(x=12.0, y=10.0), Food(x=18.0, y=10.0)])
    assert w.observe().food_front == pytest.approx(0.5)

    w = make_world(foods=[Food(x=10.0, y=13.0), Food(x=10.0, y=8.0)])  # left far, right near
    obs = w.observe()
    assert obs.food_right == pytest.approx(0.5)
    assert obs.food_left == 0.0


def test_hunger_grows_and_saturates() -> None:
    w = make_world(hunger_rate=0.4)
    assert w.hunger == 0.0
    w.step(0.0, 0.0)
    assert w.hunger == pytest.approx(0.4)
    w.step(0.0, 0.0)
    w.step(0.0, 0.0)
    assert w.hunger == 1.0
    assert w.starved


def test_eating_resets_hunger_and_removes_food() -> None:
    w = make_world(foods=[Food(x=11.0, y=10.0)], hunger_rate=0.1)
    w.step(0.0, 0.0)
    obs = w.step(1.0, 1.0)  # moves to x=10.5, within reach
    assert obs.ate
    assert w.hunger == 0.0
    assert w.food_eaten == 1
    assert w.foods == []
    assert w.observe().food_front == 0.0


def test_observation_channels_include_food_and_hunger() -> None:
    ch = make_world().observe().as_channels()
    assert set(ch) == {
        "sensor_left",
        "sensor_front",
        "sensor_right",
        "food_left",
        "food_front",
        "food_right",
        "hunger",
        "target_left",
        "target_front",
        "target_right",
        "danger_left",
        "danger_front",
        "danger_right",
        "health",
        "battery",
        "dock_left",
        "dock_front",
        "dock_right",
        "ammo",
        "aim",
        "prey_left",
        "prey_front",
        "prey_right",
    }


# --- Stage 4: seeded food respawn (Plan §8 «Респаун еды») ------------------------


def test_respawn_keeps_food_count_and_is_seeded() -> None:
    def run(seed: int) -> list[tuple[float, float]]:
        w = make_world(
            foods=[Food(x=11.0, y=10.0)],
            obstacles=[Obstacle(x=15.0, y=15.0, radius=2.0)],
            respawn_food=True,
            seed=seed,
        )
        w.step(0.0, 0.0)
        w.step(1.0, 1.0)
        assert w.food_eaten == 1
        assert len(w.foods) == 1
        return [(f.x, f.y) for f in w.foods]

    assert run(1) == run(1)
    assert run(1) != run(2)


def test_respawned_food_avoids_obstacles_and_agent() -> None:
    obstacle = Obstacle(x=15.0, y=15.0, radius=2.0)
    for seed in range(20):
        w = make_world(
            foods=[Food(x=11.0, y=10.0)], obstacles=[obstacle], respawn_food=True, seed=seed
        )
        w.step(0.0, 0.0)
        w.step(1.0, 1.0)
        (f,) = w.foods
        assert math.dist((f.x, f.y), (obstacle.x, obstacle.y)) > obstacle.radius + 1.0
        assert math.dist((f.x, f.y), (w.agent.x, w.agent.y)) > 2.0
        assert 1.0 <= f.x <= 19.0
        assert 1.0 <= f.y <= 19.0


def test_no_respawn_by_default() -> None:
    w = make_world(foods=[Food(x=11.0, y=10.0)])
    w.step(0.0, 0.0)
    w.step(1.0, 1.0)
    assert w.foods == []


def test_angled_solid_lines_block_rays_and_sight() -> None:
    """Stage B11: a map that comes from the Doom engine has walls at any angle."""
    world = World(width=20.0, height=20.0, agent=AgentState(x=5.0, y=5.0, heading=0.0))
    assert world.segments == [], "generated levels use axis-aligned walls only"
    assert world.ray_distance(0.0) == pytest.approx(4.0), "nothing there: the sensor range"

    world.segments = [(7.0, 3.0, 7.0, 7.0)]  # a 4-unit line two units ahead
    assert world.ray_distance(0.0) == pytest.approx(2.0)
    assert world.ray_distance(math.radians(30)) == pytest.approx(2.0 / math.cos(math.radians(30)))
    assert world.ray_distance(math.radians(60)) == pytest.approx(4.0), "ray passes its end"
    assert world.ray_distance(math.pi) == pytest.approx(4.0), "behind the agent"
    assert not world.line_of_sight(9.0, 5.0)
    assert world.line_of_sight(6.0, 5.0)
