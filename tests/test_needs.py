"""Stage 20: needs arbitration battery > call > clean (Plan §20.3)."""

from pathlib import Path

import pytest

from doomworm.environments.sensors import IDEAL, VACUUM, SensorSuite
from doomworm.environments.worlds import build_world
from doomworm.episode import run_brain_episode
from doomworm.layer import GradientFollower, NeedsArbiter, PlannerLayer


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
    from doomworm.candidates import ScriptedBrain

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
    from doomworm.candidates import ScriptedBrain

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
    from doomworm.candidates import ScriptedBrain

    def run(reflex: bool) -> int:
        world = build_world(3002, "apartment", "clean")
        brain = ScriptedBrain(lambda _c: (1.0, 1.0), "straight")
        layer = PlannerLayer(brain, VACUUM, mode="needs", bumper_reflex=reflex)
        run_brain_episode(world, layer, 400, sensors=SensorSuite(VACUUM, seed=5))
        return world.collisions

    assert run(True) < run(False) / 3


# --- stage 22.1e: the dock trip on a camera marker with dead-reckoning odometry ---------


def test_marker_homing_trusts_any_sighting_but_the_ir_beacon_needs_strength() -> None:
    from doomworm.environments.sensors import CAR

    cam = PlannerLayer(GradientFollower(), CAR, mode="needs")
    cam.reset()
    cam.needs.state = "charge"
    cam.beacon = (0.0, 0.2, 0.0)  # marker seen 5 units away
    assert cam.homing()
    ir = PlannerLayer(GradientFollower(), VACUUM, mode="needs")
    ir.reset()
    ir.needs.state = "charge"
    ir.beacon = (0.0, 0.2, 0.0)  # the IR beacon at 5 units points through walls
    assert not ir.homing()


def test_marker_sighting_re_anchors_the_dock_estimate() -> None:
    from doomworm.environments.sensors import CAR

    layer = PlannerLayer(GradientFollower(), CAR, mode="needs")
    layer.reset()
    layer.needs.dock = (0.0, 0.0)
    base = dict.fromkeys(SensorSuite(CAR).channel_names, 0.0) | {"battery": 1.0}
    frame = base | {"odom_x": 5.0, "odom_y": 5.0, "odom_heading": 0.0, "dock_front": 0.25}
    layer.observe(frame)
    assert layer.needs.dock == pytest.approx((9.0, 5.0), abs=0.3), "4 units straight ahead"
    frame = base | {"odom_x": 5.0, "odom_y": 5.0, "odom_heading": 0.0, "dock_left": 0.5}
    layer.observe(frame)
    dx, dy = layer.needs.dock
    assert dy > 5.0 > dx - 2.0, "2 units, up and to the left"
    ir = PlannerLayer(GradientFollower(), VACUUM, mode="needs")
    ir.reset()
    ir.needs.dock = (0.0, 0.0)
    ir.observe(
        dict.fromkeys(SensorSuite(VACUUM).channel_names, 0.0) | {"battery": 1.0, "dock_front": 0.25}
    )
    assert ir.needs.dock == (0.0, 0.0), "the IR beacon does not move the estimate"


def test_marker_search_spins_near_the_estimated_dock_without_a_sighting() -> None:
    from doomworm.environments.sensors import CAR

    layer = PlannerLayer(GradientFollower(), CAR, mode="needs")
    layer.reset()
    base = dict.fromkeys(SensorSuite(CAR).channel_names, 0.0)
    frame = base | {"odom_x": 5.0, "odom_y": 5.0, "odom_heading": 0.0, "battery": 0.2}
    layer.needs.dock = (6.0, 5.0)  # "here", by dead reckoning; no marker in sight
    wheels = [layer.act(frame | {"odom_heading": 0.4 * k}) for k in range(12)]
    spins = [w for w in wheels if w[0] * w[1] < 0.0]
    assert len(spins) >= 6, f"searching = spinning in place, got {wheels}"
    assert layer.needs.state == "charge"
    # the marker appears: homing takes over, no more spinning
    seen = frame | {"odom_heading": 4.8, "dock_front": 0.3}
    w = layer.act(seen)
    assert w[0] > 0.0, f"drive at the marker, got {w}"
    assert w[1] > 0.0, f"drive at the marker, got {w}"


def test_autopilot_on_the_car_preset_docks_the_straight_driver() -> None:
    """Stage 22.1e: dead-reckoning odometry + camera marker still bring a bad driver home."""
    from doomworm.candidates import ScriptedBrain
    from doomworm.environments.sensors import CAR

    world = build_world(3001, "apartment", "clean")
    world.hunger_rate = 0.003
    straight = ScriptedBrain(lambda _c: (1.0, 1.0), "straight")
    layer = PlannerLayer(straight, CAR, mode="needs")
    trace = run_brain_episode(world, layer, 700, sensors=SensorSuite(CAR, seed=7))
    assert len(trace) == 700, "discharged"
    assert world.dockings >= 2
