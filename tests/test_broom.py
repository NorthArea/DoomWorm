"""The robot track: its task, its command line, its bench (stages J2-J3).

Until this module the suite had 275 tests and not one of them imported
`broomworm`. Splitting the two tracks quietly took `clean` out of the shared
world and all seven hardware commands out of the command line, and nothing
failed, because nothing was looking. These are the tests that would have.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import broomworm  # noqa: F401  -- importing the track registers its task and presets
from broomworm.cli import build_parser, main
from wormlab.environments.worlds import TASKS, build_world


def test_the_track_registers_its_own_task_on_import() -> None:
    """Stage J2: the dirt, dock and battery are the platform's; switching them on is ours."""
    assert "clean" in TASKS
    world = build_world(1, "apartment", "clean")
    assert world.dirt is not None, "a floor to clean"
    assert world.dock is not None, "a dock to come back to"
    assert world.dirt.coverage == 0.0, "nothing cleaned yet"
    assert world.hunger_rate > 0.0, "a battery that runs down"
    assert not world.foods, "the vacuum is not foraging"


def test_an_unknown_task_says_which_ones_exist() -> None:
    with pytest.raises(ValueError, match=r"unknown task 'sweep'.*clean, food, target"):
        build_world(1, "apartment", "sweep")


def test_a_track_cannot_register_the_same_task_twice() -> None:
    from wormlab.environments.worlds import register_task

    with pytest.raises(ValueError, match="already registered"):
        register_task("clean", lambda _w: None)


def test_every_hardware_command_is_wired() -> None:
    """The seven that were lost in the split, by name, so they cannot go again."""
    subs = [a for a in build_parser()._actions if a.__class__.__name__ == "_SubParsersAction"]
    assert set(subs[0].choices or {}) == {
        "drive",
        "compare-log",
        "selftest",
        "robustness",
        "motor-map",
        "calibrate",
        "plot-log",
    }


def test_no_command_prints_help_instead_of_guessing() -> None:
    assert main([]) == 1


def test_the_bench_maps_every_wheel_over_a_fake_robot(capsys: pytest.CaptureFixture[str]) -> None:
    """`make broom-motor-map LINK=fake`: the first thing anyone runs at the bench."""
    assert main(["motor-map", "--link", "fake", "--ms", "1"]) == 0
    printed = capsys.readouterr().out
    assert "MOTOR_FWD" in printed, "the C tables the firmware needs"
    assert "every wheel has one bit per direction" in printed


def test_the_sweep_runs_on_the_simulator_and_writes_its_table(tmp_path: Path) -> None:
    """Stage J3: the world-dependent lanes work again once the task is back."""
    out = tmp_path / "sweep.md"
    args = ["robustness", "--scripted", "roomba", "--test-seeds", "1", "--repeats", "1"]
    assert main([*args, "--steps", "20", "--params", "dropout", "--out", str(out)]) == 0
    table = out.read_text()
    assert "| parameter | value | reward |" in table
    assert "dropout" in table
    assert "coverage" in table, "the vacuum's own column, which needs the `clean` task"


def test_a_drive_log_replays_in_the_simulator(tmp_path: Path) -> None:
    log = tmp_path / "drive.jsonl"
    args = ["drive", "--scripted", "roomba", "--steps", "12", "--record", str(log)]
    assert main([*args, "--link", "sim", "--seed", "3000"]) == 0
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    meta, ticks = rows[0]["meta"], rows[1:]
    assert meta["task"] == "clean"
    assert meta["controller"] == "roomba"
    assert [r["tick"] for r in ticks] == list(range(12))
    assert "battery" in ticks[0]["channels"], "the robot's own channels are recorded"
    assert main(["compare-log", "--log", str(log), "--out", str(tmp_path / "cmp.md")]) == 0
