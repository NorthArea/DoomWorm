"""Stage 21.1: Roomba-style classical controller, the zero-learning floor (Plan §20.4)."""

from broomworm.presets import VACUUM
from broomworm.roomba import RoombaBrain
from wormlab.environments.sensors import SensorSuite
from wormlab.environments.worlds import build_world
from wormlab.episode import run_brain_episode

CLEAR = {"sensor_left": 0.0, "sensor_front": 0.0, "sensor_right": 0.0, "battery": 1.0}


def steps(brain: RoombaBrain, channels: dict[str, float], n: int) -> list[tuple[float, float]]:
    return [brain.act(channels) for _ in range(n)]


def test_spiral_opens_up_then_bump_backs_off_and_turns_away() -> None:
    b = RoombaBrain()
    b.reset()
    first, later = b.act(CLEAR), steps(b, CLEAR, 100)[-1]
    assert first[0] > first[1], "spiral: outer wheel faster"
    assert later[1] > first[1], "spiral radius grows"
    escape = steps(b, CLEAR | {"bumper_left": 1.0}, 1) + steps(b, CLEAR, 12)
    assert all(left < 0 and right < 0 for left, right in escape[:3]), "backs off first"
    turning = [w for w in escape[3:] if w[0] != w[1]]
    assert turning
    assert all(left > right for left, right in turning), "turns right, away from a left bump"


def test_cliff_is_treated_like_a_bump() -> None:
    b = RoombaBrain()
    b.reset()
    left, right = b.act(CLEAR | {"cliff_right": 1.0})
    assert left < 0
    assert right < 0


def test_wall_following_keeps_the_wall_on_the_right() -> None:
    b = RoombaBrain()
    b.reset()
    steps(b, CLEAR | {"bumper_right": 1.0}, 1)
    steps(b, CLEAR, 30)  # escape done, now following
    assert b.state == "follow"
    near, far = b.act(CLEAR | {"wall_right": 0.9}), b.act(CLEAR | {"wall_right": 0.1})
    assert near[0] < near[1], "too close: steer left"
    assert far[0] > far[1], "too far: steer right"


def test_low_battery_homes_on_the_beacon_and_parks_while_charging() -> None:
    b = RoombaBrain()
    b.reset()
    low = CLEAR | {"battery": 0.3}
    assert b.act(low | {"dock_left": 0.3}) == b.act(low | {"dock_left": 0.3})
    left, right = b.act(low | {"dock_left": 0.3})
    assert left < right, "beacon on the left: turn left"
    left, right = b.act(low | {"dock_right": 0.3})
    assert left > right
    left, right = b.act(low | {"dock_front": 0.5})
    assert left == right > 0
    assert b.act(low | {"dock_front": 1.0, "battery": 0.32}) == (0.0, 0.0), "charging: park"
    assert b.act(low | {"dock_front": 1.0, "battery": 0.96}) != (0.0, 0.0), "full: leave"
    assert b.state == "spiral"


def test_cleans_and_recharges_on_an_unseen_apartment_with_vacuum_sensors() -> None:
    """Benchmark conditions (drain 0.002): the floor must at least come back once."""
    world = build_world(3002, "apartment", "clean")
    brain = RoombaBrain()
    trace = run_brain_episode(world, brain, 800, sensors=SensorSuite(VACUUM, seed=3002000))
    assert len(trace) == 800, "discharged"
    assert world.dockings >= 2, "returned to the dock at least once"
    assert world.coverage > 0.1
