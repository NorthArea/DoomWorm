"""Stage 1: 2D world, differential drive, 3 distance sensors (Plan §7)."""

import math

import pytest

from doomworm.environments.simple_2d import AgentState, Obstacle, World

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
    w.step(5.0, -3.0)  # clamped to (1, 0): turn right at half speed
    assert math.dist((w.agent.x, w.agent.y), (10.0, 10.0)) == pytest.approx(0.25)
    assert w.agent.heading == pytest.approx(-0.5)


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
    assert set(ch) == {"sensor_left", "sensor_front", "sensor_right"}
    assert all(isinstance(v, float) for v in ch.values())
