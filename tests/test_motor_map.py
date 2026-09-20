"""Stage 23.2 bench tool: the shift-register bit map recovered through the protocol."""

from broomworm.hardware import Calibration, FakeRobot, MotorMapReport, motor_map, serve_fake
from broomworm.hardware.fake_robot import PROBE_MAP
from broomworm.hardware.link import LineLink, SimLink
from broomworm.presets import CAR
from wormlab.environments.worlds import build_world


def fake_link() -> LineLink:
    sim = SimLink(build_world(3001, "apartment", "clean"), CAR, sensor_seed=0)
    return serve_fake(FakeRobot(sim, Calibration.for_preset("car")))


def test_probe_command_answers_with_a_reading_and_the_wheel() -> None:
    link = fake_link()
    try:
        reply = link.command({"cmd": "probe", "bits": 128, "duty": 0.5, "ms": 300})
    finally:
        link.close()
    assert "ranges_m" in reply
    assert (reply["probe"], reply["probe_wheel"], reply["probe_dir"]) == (128, "FL", "f")


def test_motor_map_recovers_the_fake_tables() -> None:
    link = fake_link()
    try:
        report = motor_map(link, duty=0.5, ms=200)
    finally:
        link.close()
    assert not report.problems
    fwd, bwd = report.tables()
    assert fwd == [128, 32, 8, 2]
    assert bwd == [64, 16, 4, 1]
    assert "MOTOR_FWD[4] = {128, 32, 8, 2};" in report.c_tables()
    assert "| 128 | FL | f |" in report.markdown()
    assert set(report.observed) == set(PROBE_MAP)


def test_report_flags_missing_and_shared_bits() -> None:
    r = MotorMapReport(
        observed={128: ("FL", "f"), 64: ("FL", "b"), 32: ("FR", "f"), 16: ("FR", "f")}
    )
    r.check()
    text = "\n".join(r.problems)
    assert "no bit drives FR backward" in text
    assert "no bit drives RL forward" in text
    multi = MotorMapReport(observed={8: ("multi", "?")})
    multi.check()
    assert any("more than one wheel" in p for p in multi.problems)


def test_motor_map_cli_on_the_fake_car(capsys) -> None:  # type: ignore[no-untyped-def]
    from broomworm.cli import main

    assert main(["motor-map", "--link", "fake", "--ms", "100"]) == 0
    out = capsys.readouterr().out
    assert "MOTOR_FWD" in out
    assert "OK" in out
