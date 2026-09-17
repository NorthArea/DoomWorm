"""Stage 22.1: one robot link for the simulator and the machine, teleop, record, compare."""

from __future__ import annotations

import io
import json
import math
import socket
import threading
from pathlib import Path
from typing import TextIO, cast

import pytest

from doomworm.candidates import ScriptedBrain
from doomworm.environments.sensors import IDEAL, VACUUM, SensorSuite
from doomworm.environments.worlds import build_world
from doomworm.episode import run_brain_episode
from doomworm.hardware import (
    Calibration,
    FakeRobot,
    LineLink,
    SimLink,
    Teleop,
    compare_logs,
    drive,
    read_drive_log,
    replay_in_sim,
)
from doomworm.hardware.calibration import RawReading
from doomworm.layer import GradientFollower, PlannerLayer

# --- calibration -----------------------------------------------------------------


def test_calibration_units_round_trip() -> None:
    cal = Calibration(sensors=VACUUM)
    assert cal.ray_range_m == pytest.approx(VACUUM.ray_range * cal.unit_m)
    # proximity is 1 - d / range, clipped; None = no echo = nothing in range
    assert cal.proximity(0.0, cal.ray_range_m) == 1.0
    assert cal.proximity(cal.ray_range_m, cal.ray_range_m) == 0.0
    assert cal.proximity(None, cal.ray_range_m) == 0.0
    assert cal.proximity(cal.ray_range_m / 2, cal.ray_range_m) == pytest.approx(0.5)
    # metres <-> world units
    assert cal.to_units(cal.unit_m * 3.0) == pytest.approx(3.0)
    assert cal.to_metres(3.0) == pytest.approx(cal.unit_m * 3.0)


def test_calibration_raw_to_channels_matches_the_suite_layout() -> None:
    cal = Calibration(sensors=VACUUM)
    raw = RawReading(
        ranges_m=[cal.ray_range_m / 2, None, 0.0, cal.ray_range_m, 0.25 * cal.ray_range_m],
        bumper=(1, 0),
        cliff=(0, 1),
        wall_m=cal.wall_range_m / 4,
        odom_m=(cal.unit_m * 2.0, cal.unit_m * -1.0),
        odom_rad=0.5,
        gyro_rad=0.4,
        battery=0.8,
        charging=0,
        dock=(0.0, 0.3, 0.0),
    )
    ch = cal.channels(raw)
    suite = SensorSuite(VACUUM)
    assert list(ch) == suite.channel_names, "same channels, same order as the simulator"
    assert ch["range_0"] == pytest.approx(0.5)
    assert ch["range_1"] == 0.0
    assert ch["range_2"] == 1.0
    assert ch["range_3"] == 0.0
    assert ch["range_4"] == pytest.approx(0.75)
    # sectors: rays at 60, 30 (left), 0 (front), -30, -60 (right)
    assert ch["sensor_left"] == pytest.approx(0.5)
    assert ch["sensor_front"] == 1.0
    assert ch["sensor_right"] == pytest.approx(0.75)
    assert (ch["bumper_left"], ch["bumper_right"]) == (1.0, 0.0)
    assert (ch["cliff_left"], ch["cliff_right"]) == (0.0, 1.0)
    assert ch["wall_right"] == pytest.approx(0.75)
    assert (ch["odom_x"], ch["odom_y"]) == pytest.approx((2.0, -1.0))
    assert ch["odom_heading"] == 0.5
    assert ch["gyro_heading"] == 0.4
    assert ch["battery"] == pytest.approx(0.8)
    assert ch["hunger"] == pytest.approx(0.2)
    assert ch["health"] == 1.0
    assert ch["dock_front"] == pytest.approx(0.3)
    assert ch["food_front"] == 0.0


def test_calibration_rejects_wrong_ray_count() -> None:
    cal = Calibration(sensors=VACUUM)
    raw = RawReading(ranges_m=[1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="5 rays"):
        cal.channels(raw)


# --- simulator behind the link -----------------------------------------------------


def test_sim_link_reproduces_the_episode_loop() -> None:
    """Driving a brain through SimLink is the same closed loop as run_brain_episode."""
    brain = PlannerLayer(GradientFollower(), VACUUM, mode="needs")
    world_a = build_world(3000, "apartment", "clean")
    trace = run_brain_episode(world_a, brain, 60, sensors=SensorSuite(VACUUM, seed=7))

    link = SimLink(build_world(3000, "apartment", "clean"), VACUUM, sensor_seed=7)
    rows = drive(link, brain, steps=60)
    assert len(rows) == len(trace)
    # Since stage 24 the loop logs the wheels its body produced, which round-trip
    # through the intent and can differ from the brain's own pair by one ULP; the
    # motion itself is bit-identical, which the pose check below pins down.
    flat = [v for r in rows for v in r.wheels]
    assert flat == pytest.approx([v for t in trace for v in t.motors], abs=1e-12)
    assert rows[-1].truth is not None, "the simulator link knows the true pose"
    assert link.truth()[:2] == pytest.approx((trace[-1].x, trace[-1].y))


def test_sim_link_first_reading_is_at_rest_and_warm() -> None:
    link = SimLink(build_world(3000, "apartment", "clean"), VACUUM, sensor_seed=1)
    first = link.reset()
    assert set(first) == set(SensorSuite(VACUUM).channel_names)
    assert first["battery"] == 1.0
    assert first["odom_x"] == pytest.approx(link.world.agent.x)
    assert first["range_2"] >= 0.0


# --- the wire: JSON lines between host and robot --------------------------------------


def _serve_fake_robot(robot: FakeRobot) -> tuple[LineLink, threading.Thread]:
    host_sock, robot_sock = socket.socketpair()
    robot_file = robot_sock.makefile("rw", encoding="utf-8", newline="\n")
    thread = threading.Thread(target=robot.serve, args=(robot_file, robot_file), daemon=True)
    thread.start()
    host_file = host_sock.makefile("rw", encoding="utf-8", newline="\n")
    link = LineLink(host_file, host_file, robot.calibration, name="socketpair", transport=host_sock)
    robot_sock.close()  # the robot side lives on in its file object
    return link, thread


def test_line_link_talks_to_the_fake_robot_like_the_simulator() -> None:
    """Same brain, same seed: the wire protocol changes nothing but the transport."""
    cal = Calibration(sensors=VACUUM)
    brain = PlannerLayer(GradientFollower(), VACUUM, mode="needs")

    direct = SimLink(build_world(3001, "apartment", "clean"), VACUUM, sensor_seed=3)
    rows_direct = drive(direct, brain, steps=40)

    robot = FakeRobot(SimLink(build_world(3001, "apartment", "clean"), VACUUM, sensor_seed=3), cal)
    link, thread = _serve_fake_robot(robot)
    rows_wire = drive(link, brain, steps=40)
    link.close()
    thread.join(timeout=5)

    assert [r.wheels for r in rows_wire] == pytest.approx([r.wheels for r in rows_direct])
    for a, b in zip(rows_wire, rows_direct, strict=True):
        for key, value in b.channels.items():
            assert a.channels[key] == pytest.approx(value, abs=1e-6), key
        assert a.raw is not None
        assert "ranges_m" in a.raw


def test_line_link_checks_the_handshake() -> None:
    reader = io.StringIO(json.dumps({"hello": "not a reading"}) + "\n")
    writer = io.StringIO()
    link = LineLink(reader, writer, Calibration(sensors=VACUUM))
    with pytest.raises(ValueError, match="reading"):
        link.reset()
    assert json.loads(writer.getvalue().splitlines()[0]) == {"cmd": "reset"}


def test_line_link_sends_clipped_wheel_commands() -> None:
    cal = Calibration(sensors=VACUUM)
    reading = json.dumps(RawReading(ranges_m=[None] * 5).to_dict()) + "\n"
    reader = io.StringIO(reading * 2)
    writer = io.StringIO()
    link = LineLink(reader, writer, cal)
    link.reset()
    link.step(2.0, -3.0)
    lines = [json.loads(line) for line in writer.getvalue().splitlines()]
    assert lines == [{"cmd": "reset"}, {"cmd": "drive", "left": 1.0, "right": -1.0}]


# --- teleop and the drive log -------------------------------------------------------------


def test_teleop_parses_keys_and_pairs_and_stops_on_q() -> None:
    tele = Teleop(io.StringIO("w\na\n0.3 -0.3\n\nx\nq\nw\n"))
    tele.reset()
    assert tele.act({}) == (1.0, 1.0)
    assert tele.act({}) == (-0.6, 0.6)
    assert tele.act({}) == (0.3, -0.3)
    assert tele.act({}) == (0.3, -0.3), "blank line repeats the last command"
    assert tele.act({}) == (0.0, 0.0)
    assert tele.act({}) == (0.0, 0.0)
    assert tele.stopped


def test_drive_records_a_replayable_log(tmp_path: Path) -> None:
    link = SimLink(build_world(3002, "apartment", "clean"), VACUUM, sensor_seed=5)
    log = tmp_path / "drive.jsonl"
    rows = drive(link, Teleop(io.StringIO("w\nw\nd\nw\n")), steps=50, log=log, meta={"who": "t"})
    assert len(rows) == 4, "teleop ends the drive when its input runs out"
    meta, rows_back = read_drive_log(log)
    assert meta["who"] == "t"
    assert (meta["link"], meta["sensors"]) == ("sim", "vacuum")
    assert [r.wheels for r in rows_back] == [r.wheels for r in rows]
    assert rows_back[2].channels == rows[2].channels
    assert rows_back[0].truth == pytest.approx(rows[0].truth)


def test_replay_of_a_sim_log_is_exact_and_a_preset_change_is_visible(tmp_path: Path) -> None:
    """The sim-vs-real check: replay the recorded wheels in the simulator, compare channels."""
    link = SimLink(build_world(3003, "apartment", "clean"), VACUUM, sensor_seed=9, seed=3003)
    log = tmp_path / "drive.jsonl"
    drive(link, ScriptedBrain(lambda _: (1.0, 0.8)), steps=80, log=log)
    meta, rows = read_drive_log(log)

    same = replay_in_sim(meta, rows)
    report = compare_logs(rows, same)
    assert report.ticks == 80
    assert report.max_abs("range_2") == 0.0
    assert report.rmse("odom_x") == 0.0
    assert report.bumper_agreement == 1.0
    assert report.pose_error_m(Calibration(sensors=VACUUM)) == 0.0

    ideal = replay_in_sim(meta | {"sensors": "ideal"}, rows)
    report_ideal = compare_logs(rows, ideal)
    assert report_ideal.rmse("odom_x") > 0.0, "odometry noise of the vacuum preset shows up"
    assert report_ideal.rmse("range_2") > 0.0
    text = report_ideal.markdown()
    assert "range_2" in text
    assert "odom_x" in text
    assert set(report_ideal.channels) < set(rows[0].channels), "only common channels"
    assert "range_3" not in report_ideal.channels, "the ideal preset has three rays"


def test_compare_handles_different_lengths_and_missing_truth() -> None:
    link = SimLink(build_world(3003, "apartment", "clean"), IDEAL, sensor_seed=0)
    rows = drive(link, ScriptedBrain(lambda _: (0.5, 0.5)), steps=30)
    short = [r.__class__(r.tick, r.wheels, r.channels, None, None) for r in rows[:20]]
    report = compare_logs(rows, short)
    assert report.ticks == 20
    assert report.pose_error_m(Calibration()) is None
    assert math.isfinite(report.rmse("odom_heading"))


# --- CLI ---------------------------------------------------------------------------------


def test_cli_drive_and_compare_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from doomworm.cli import main

    log = tmp_path / "drive.jsonl"
    monkeypatch.setattr("sys.stdin", io.StringIO("w\nw\nw\na\nw\n"))
    assert main(["drive", "--teleop", "--seed", "3004", "--steps", "10", "--record", str(log)]) == 0
    meta, rows = read_drive_log(log)
    assert len(rows) == 5
    assert meta["seed"] == 3004
    out = tmp_path / "compare.md"
    assert main(["compare-log", "--log", str(log), "--out", str(out)]) == 0
    assert "bumper agreement" in out.read_text()

    brain = Path("docs/brains/a2/worm_from_worm_evolved_random.json")
    args = ["drive", "--brain", str(brain), "--planner", "needs", "--seed", "3004", "--steps", "5"]
    assert main([*args, "--record", str(tmp_path / "brain.jsonl")]) == 0
    with pytest.raises(SystemExit):
        main(["drive", "--seed", "1"])


# --- stage 22 car platform --------------------------------------------------------------


def test_car_calibration_scale_and_speed() -> None:
    cal = Calibration.for_preset("car")
    assert cal.sensors.name == "car"
    assert cal.unit_m == pytest.approx(0.2)
    assert cal.speed_mps == pytest.approx(0.4)
    assert Calibration.for_preset("vacuum").unit_m == pytest.approx(0.33)


def test_line_link_integrates_odometry_when_the_machine_has_no_encoders() -> None:
    cal = Calibration.for_preset("car")
    frame = json.dumps(RawReading(ranges_m=[None] * 3, odom_m=None).to_dict()) + "\n"
    link = LineLink(io.StringIO(frame * 12), io.StringIO(), cal)
    first = link.reset()
    assert (first["odom_x"], first["odom_y"]) == (0.0, 0.0)
    ch = first
    for _ in range(10):
        ch = link.step(1.0, 1.0)
    assert ch["odom_x"] == pytest.approx(10 * cal.speed)
    assert ch["gyro_heading"] == ch["odom_heading"], "no IMU on the car"
    ch = link.step(-1.0, 1.0)
    assert ch["odom_heading"] == pytest.approx(2.0 / cal.wheel_base * cal.speed)


def test_fake_robot_serves_the_car_preset() -> None:
    from doomworm.environments.sensors import CAR

    cal = Calibration.for_preset("car")
    brain = PlannerLayer(GradientFollower(), CAR, mode="needs")
    direct = SimLink(build_world(3001, "apartment", "clean"), CAR, sensor_seed=3)
    rows_direct = drive(direct, brain, steps=30)
    robot = FakeRobot(SimLink(build_world(3001, "apartment", "clean"), CAR, sensor_seed=3), cal)
    link, thread = _serve_fake_robot(robot)
    rows_wire = drive(link, brain, steps=30)
    link.close()
    thread.join(timeout=5)
    assert [r.wheels for r in rows_wire] == pytest.approx([r.wheels for r in rows_direct])
    # every channel, not just the wheels: the wire rebuilds bumper and wall from the
    # raw reading, and a machine that serves fewer keys than the suite is the day-one bug
    for wire, direct_row in zip(rows_wire, rows_direct, strict=True):
        assert set(wire.channels) == set(direct_row.channels)
        for key, value in direct_row.channels.items():
            assert wire.channels[key] == pytest.approx(value, abs=1e-9), key


# --- stage 22.2 preparation: room files, self-test, calibration, pictures -------------


def test_room_file_in_metres_builds_a_world_and_round_trips() -> None:
    from doomworm.hardware import Room, load_room, room_world

    room = load_room("data/rooms/example_room.json")
    assert (room.width, room.height) == pytest.approx((15.0, 20.0)), "3 x 4 m at 0.2 m/u"
    assert room.start[2] == pytest.approx(math.pi / 2)
    world = room_world(room, seed=1)
    assert world.dock is not None
    assert world.dirt is not None
    assert world.agent.x == pytest.approx(2.5)
    assert len(world.walls) == 2
    again = Room.from_dict(room.to_dict())
    assert again == room


def test_build_world_accepts_a_room_and_drive_log_carries_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from doomworm.cli import main
    from doomworm.environments.worlds import build_world

    world = build_world(0, "room:data/rooms/example_room.json", "clean")
    assert world.width == pytest.approx(15.0)
    log = tmp_path / "room.jsonl"
    args = ["drive", "--teleop", "--sensors", "car", "--room", "data/rooms/example_room.json"]
    monkeypatch.setattr("sys.stdin", io.StringIO("w\nw\nw\n"))
    assert main([*args, "--steps", "5", "--record", str(log)]) == 0
    meta, rows = read_drive_log(log)
    assert meta["room"]["width"] == pytest.approx(15.0)
    # replay needs no file: the room travels inside the log
    same = replay_in_sim(meta, rows)
    assert compare_logs(rows, same).rmse("odom_x") == 0.0
    assert main(["compare-log", "--log", str(log), "--room", "data/rooms/example_room.json"]) == 0
    assert main(["plot-log", "--log", str(log)]) == 0
    assert log.with_suffix(".png").exists()


def test_selftest_passes_on_the_simulator_and_flags_dead_odometry() -> None:
    from doomworm.environments.sensors import CAR
    from doomworm.hardware import selftest

    link = SimLink(build_world(3001, "apartment", "clean"), CAR, sensor_seed=1)
    report = selftest(link, rest_frames=5)
    assert report.ok, report.problems
    assert report.heading_left_spin > 0 > report.heading_right_spin
    assert report.forward_units > 0
    assert len(report.roundtrip_ms) == 5
    assert "OK" in report.markdown()

    class Frozen(SimLink):
        def step(self, left: float, right: float) -> dict[str, float]:
            return super().step(0.0, 0.0)  # wheels wired to nothing

    frozen = Frozen(build_world(3001, "apartment", "clean"), CAR, sensor_seed=1)
    bad = selftest(frozen, rest_frames=2)
    assert not bad.ok
    assert any("heading" in p for p in bad.problems)


def test_calibrate_recovers_the_simulator_scale(tmp_path: Path) -> None:
    from doomworm.hardware import calibrate, load_calibration, save_calibration

    # a "car" whose true scale is 0.25 m/u: an empty sim world, measured with a tape
    world = build_world(3001, "fixed", "food")
    world.walls, world.obstacles = [], []
    link = SimLink(world, "ideal", 0)
    base = Calibration.for_preset("car")
    answers = iter(["1.0", "45"])  # 20 ticks * 0.2 u = 4 u = 1.0 m -> 0.25 m/u; spin 45 deg
    said: list[str] = []
    cal = calibrate(link, base, ticks=20, ask=lambda _: next(answers), say=said.append)
    assert cal.unit_m == pytest.approx(0.25)
    assert cal.wheel_base == pytest.approx(8.0 / math.radians(45), rel=1e-3)
    assert cal.sensors.name == "car"
    path = save_calibration(cal, tmp_path / "cal.json")
    back = load_calibration(path)
    assert back.unit_m == pytest.approx(0.25)
    assert back.wheel_base == pytest.approx(cal.wheel_base)
    assert back.sensors.name == "car"
    assert any("measured" in line for line in said)


def test_cli_selftest_and_calibrate_on_the_simulator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from doomworm.cli import main

    out = tmp_path / "selftest.md"
    st = ["selftest", "--sensors", "car", "--seed", "3001", "--frames", "3"]
    assert main([*st, "--out", str(out)]) == 0
    assert "OK" in out.read_text()
    answers = iter(["0.8", "60"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    cal = tmp_path / "cal.json"
    cl = ["calibrate", "--sensors", "car", "--seed", "3001", "--ticks", "20"]
    assert main([*cl, "--out", str(cal)]) == 0
    assert json.loads(cal.read_text())["unit_m"] == pytest.approx(0.2)
    # a measured file feeds the tcp link; on the sim link it is accepted and ignored
    assert main([*st, "--calibration", str(cal)]) == 0


def test_car_frame_carries_every_channel_the_car_preset_defines() -> None:
    """Stage 22.2: the real link must serve the same keys as the simulator's suite.

    The car has no bumper: contact is derived from the rays (``proximity_bumper``).
    Without it the day-one self-test reports missing channels and the layer's bumper
    reflex reads a permanent zero on the machine.
    """
    from doomworm.environments.sensors import CAR

    cal = Calibration.for_preset("car")
    raw = RawReading(
        ranges_m=[0.10, 0.12, 0.30],
        bumper=(0, 0),
        cliff=(0, 0),
        wall_m=0.10,
        odom_m=None,
        odom_rad=0.0,
        gyro_rad=0.0,
        battery=1.0,
        charging=False,
        dock=(0.0, 0.0, 0.0),
    )
    channels = cal.channels(raw)
    assert set(SensorSuite(CAR).channel_names) <= set(channels)
    # 0.10 m = 0.5 u is inside the 0.7 u proximity threshold, on the front and left rays
    assert channels["bumper_left"] == 1.0
    assert channels["bumper_right"] == 1.0


def test_car_wall_sensor_is_binary_on_the_machine_too() -> None:
    """The IR module is a binary obstacle sensor (0.75 u), not a graded proximity."""
    cal = Calibration.for_preset("car")
    assert cal.wall_channel(0.10) == 1.0, "0.5 u is inside the 0.75 u trip point"
    assert cal.wall_channel(0.20) == 0.0, "1.0 u is outside it"
    assert cal.wall_channel(None) == 0.0, "no echo = nothing there"


def test_selftest_says_when_odometry_is_only_dead_reckoning() -> None:
    """Stage 22.2: on a machine without encoders the wiggle check tests the host, not the car."""
    from doomworm.environments.sensors import CAR
    from doomworm.hardware import selftest

    world = build_world(3001, "apartment", "clean")
    report = selftest(SimLink(world, CAR, sensor_seed=3), rest_frames=3)
    assert report.ok, "a dead-reckoned link is not a failure"
    assert any("dead reckoning" in note for note in report.notes)
    assert "NOTE:" in report.markdown()

    vacuum = selftest(SimLink(build_world(3001, "apartment", "clean"), VACUUM), rest_frames=3)
    assert vacuum.notes == [], "encoders: the check means what it says"


def test_line_link_reports_a_robot_error_and_a_silent_machine() -> None:
    """Stage 22.2: the firmware answers every command, and a stall must not look like a hang."""

    class Silent:
        """A machine that stops answering mid-drive."""

        def readline(self) -> str:
            raise TimeoutError("socket timed out")

    cal = Calibration.for_preset("car")
    error_line = json.dumps({"error": "unknown command"}) + "\n"
    link = LineLink(io.StringIO(error_line), io.StringIO(), cal)
    with pytest.raises(ValueError, match="rejected 'reset': unknown command"):
        link.reset()

    stalled = LineLink(cast("TextIO", Silent()), io.StringIO(), cal)
    with pytest.raises(ConnectionError, match="no reply to 'reset'"):
        stalled.reset()
