"""Stage 20: needs arbitration battery > call > clean (Plan §20.3)."""

from pathlib import Path

import pytest

from doomworm.brains import GradientFollower, NeedsArbiter, PlannerLayer
from doomworm.environments.sensors import IDEAL, VACUUM, SensorSuite
from doomworm.episode import run_brain_episode
from doomworm.worlds import build_world


def test_arbiter_priorities_and_hysteresis() -> None:
    a = NeedsArbiter(low=0.4, full=0.95, reach=1.0)
    assert a.decide((5.0, 5.0, 0.0), 1.0) == ("clean", None)
    assert a.dock == (5.0, 5.0), "dock = where the episode started"
    a.request_call(9.0, 9.0)
    assert a.decide((5.0, 5.0, 0.0), 0.8) == ("call", (9.0, 9.0))
    assert a.decide((5.0, 5.0, 0.0), 0.4) == ("charge", (5.0, 5.0)), "battery beats the call"
    assert a.decide((5.0, 5.0, 0.0), 0.5) == ("charge", (5.0, 5.0)), "keeps charging until full"
    assert a.decide((5.0, 5.0, 0.0), 0.96) == ("call", (9.0, 9.0))
    assert a.decide((8.5, 8.5, 0.0), 0.96) == ("clean", None), "call reached within 1.0"
    assert a.call is None


def test_trip_cost_moves_the_battery_threshold() -> None:
    a = NeedsArbiter()
    pose = (5.0, 5.0, 0.0)
    a.decide(pose, 1.0)  # dock = (5, 5)
    assert a.decide(pose, 0.5, trip=0.05)[0] == "clean"
    assert a.decide(pose, 0.5, trip=0.15)[0] == "charge", "far from the dock: leave earlier"


def test_needs_mode_returns_to_dock_and_survives() -> None:
    world = build_world(3001, "apartment", "clean")
    world.hunger_rate = 0.003  # drains in 333 ticks so the test sees a recharge
    layer = PlannerLayer(GradientFollower(), IDEAL, mode="needs")
    trace = run_brain_episode(world, layer, 700, sensors=SensorSuite(IDEAL))
    assert len(trace) == 700, "did not discharge"
    assert world.dockings >= 2, "left the dock to clean and came back to charge"
    assert min(r.battery for r in trace) < 0.45
    assert max(r.battery for r in trace[300:]) > 0.9, "recharged"
    assert world.coverage > 0.05


def test_needs_mode_survives_with_drifting_odometry() -> None:
    """Vacuum preset: the dock is found by its beacon, not by dead reckoning."""
    world = build_world(3001, "apartment", "clean")
    world.hunger_rate = 0.003
    layer = PlannerLayer(GradientFollower(), VACUUM, mode="needs")
    trace = run_brain_episode(world, layer, 700, sensors=SensorSuite(VACUUM, seed=7))
    assert len(trace) == 700, "discharged"
    assert world.dockings >= 2
    assert max(r.battery for r in trace[300:]) > 0.9, "recharged"
    assert world.coverage > 0.05


def test_beacon_overrides_waypoint_and_charging_parks() -> None:
    layer = PlannerLayer(GradientFollower(), IDEAL, mode="needs")
    layer.reset()
    base = {"odom_x": 5.0, "odom_y": 5.0, "odom_heading": 0.0, "battery": 0.3}
    layer.act(base)  # battery low -> state charge, dock = (5, 5), beacon out of range
    assert layer.needs.state == "charge"
    assert not layer.homing()
    layer.act(base | {"dock_right": 0.4})  # 2.5 units away: too far to trust the bearing
    assert not layer.homing()
    layer.act(base | {"dock_right": 0.6})
    assert layer.homing()
    assert layer.gradient() == {"target_left": 0.0, "target_front": 0.0, "target_right": 0.6}
    assert not layer.parked()
    layer.act(base | {"odom_x": 6.0, "battery": 0.32, "dock_front": 1.0})  # battery rose
    assert layer.parked()
    assert layer.needs.dock == (6.0, 5.0), "dock re-anchored where charging happens"


def test_call_takes_the_robot_to_a_point() -> None:
    world = build_world(3002, "apartment", "clean")
    layer = PlannerLayer(GradientFollower(), IDEAL, mode="needs", replan_every=3)
    start_room = world.room_index(world.agent.x, world.agent.y)
    assert start_room is not None
    x0, y0, x1, y1 = world.rooms[start_room ^ 1]  # horizontal neighbour in the 2x2 grid
    layer.reset()
    layer.call((x0 + x1) / 2, (y0 + y1) / 2)
    trace = run_brain_episode(world, layer, 500, sensors=SensorSuite(IDEAL))
    visited = {world.room_index(r.x, r.y) for r in trace}
    assert (start_room ^ 1) in visited
    assert layer.needs.call is None, "call cleared on arrival"


def test_benchmark_cli_driver_and_needs_row_name(tmp_path: Path) -> None:
    from doomworm.cli import main

    args = [
        "benchmark",
        "--scripted",
        "follower",
        "--planner",
        "needs",
        "--test-seeds",
        "1",
        "--repeats",
        "1",
    ]
    assert main([*args, "--steps", "20", "--out-dir", str(tmp_path)]) == 0
    assert (tmp_path / "driver_follower+needs.json").exists()
    # the ideal preset is noiseless, not sensorless: the planner still gets odometry
    ideal = [*args, "--sensors", "ideal", "--steps", "20", "--out-dir", str(tmp_path / "ideal")]
    assert main(ideal) == 0
    assert "driver_follower+needs" in (tmp_path / "leaderboard.md").read_text()
    with pytest.raises(SystemExit):
        main(["benchmark", "--out-dir", str(tmp_path)])


def test_call_works_with_drifting_odometry() -> None:
    world = build_world(3002, "apartment", "clean")
    layer = PlannerLayer(GradientFollower(), VACUUM, mode="needs", replan_every=3)
    start_room = world.room_index(world.agent.x, world.agent.y)
    assert start_room is not None
    x0, y0, x1, y1 = world.rooms[start_room ^ 1]
    layer.reset()
    layer.call((x0 + x1) / 2, (y0 + y1) / 2)
    trace = run_brain_episode(world, layer, 500, sensors=SensorSuite(VACUUM, seed=3))
    assert (start_room ^ 1) in {world.room_index(r.x, r.y) for r in trace}
    assert layer.needs.call is None, "call cleared on arrival"


def test_dock_autopilot_brings_a_bad_driver_home() -> None:
    """Stage 21.8: the return to dock is the layer's routine, whatever the brain does."""
    from doomworm.brains import ScriptedBrain

    world = build_world(3001, "apartment", "clean")
    world.hunger_rate = 0.003
    straight = ScriptedBrain(lambda _c: (1.0, 1.0), "straight")  # drives into walls forever
    layer = PlannerLayer(straight, VACUUM, mode="needs")
    trace = run_brain_episode(world, layer, 700, sensors=SensorSuite(VACUUM, seed=7))
    assert len(trace) == 700, "discharged"
    assert world.dockings >= 2
    assert max(r.battery for r in trace[300:]) > 0.9, "recharged"
    off = PlannerLayer(straight, VACUUM, mode="needs", dock_autopilot=False, bumper_reflex=False)
    world2 = build_world(3001, "apartment", "clean")
    world2.hunger_rate = 0.003
    trace2 = run_brain_episode(world2, off, 700, sensors=SensorSuite(VACUUM, seed=7))
    assert len(trace2) < 700, "without the autopilot the straight driver dies"


def test_bumper_reflex_backs_off_and_turns_away() -> None:
    """Stage 21.9: contact is handled by the layer before the brain sees anything."""
    from doomworm.brains import ScriptedBrain

    calls: list[int] = []

    def straight(_c: object) -> tuple[float, float]:
        calls.append(1)
        return 1.0, 1.0

    layer = PlannerLayer(ScriptedBrain(straight, "straight"), IDEAL, mode="coverage")
    layer.reset()
    base = {"odom_x": 5.0, "odom_y": 5.0, "odom_heading": 0.0, "battery": 1.0}
    assert layer.act(base) == (1.0, 1.0)
    first = layer.act(base | {"bumper_left": 1.0})
    assert first == (-0.6, -0.6), "backs off"
    n = len(calls)
    wheels = [layer.act(base) for _ in range(8)]
    assert len(calls) == n, "the brain is not consulted during the manoeuvre"
    assert wheels[:2] == [(-0.6, -0.6)] * 2
    assert all(w == (1.0, -1.0) for w in wheels[2:8]), "spins right, away from a left bump"
    assert layer.act(base) == (1.0, 1.0), "manoeuvre over, the brain drives again"


def test_bumper_reflex_cuts_collisions_of_a_bad_driver() -> None:
    from doomworm.brains import ScriptedBrain

    def run(reflex: bool) -> int:
        world = build_world(3002, "apartment", "clean")
        brain = ScriptedBrain(lambda _c: (1.0, 1.0), "straight")
        layer = PlannerLayer(brain, VACUUM, mode="needs", bumper_reflex=reflex)
        run_brain_episode(world, layer, 400, sensors=SensorSuite(VACUUM, seed=5))
        return world.collisions

    assert run(True) < run(False) / 3
