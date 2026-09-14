"""Stage 1 acceptance: the agent drives, reacts to sensors, turns (Plan §7)."""

import math

from doomworm.experiments.obstacle_agent import SENSOR_THRESHOLD, build_scenario, run_episode


def test_agent_drives_reacts_and_turns() -> None:
    trace = run_episode(*build_scenario(), steps=150)

    start, end = trace[0], trace[-1]
    assert math.dist((start.x, start.y), (end.x, end.y)) > 1.0, "agent moved"

    first_trigger = next(i for i, r in enumerate(trace) if max(r.sensors) >= SENSOR_THRESHOLD)
    headings_before = {round(r.heading, 9) for r in trace[: first_trigger + 1]}
    assert headings_before == {trace[0].heading}, "no turning before a sensor fires"

    assert any(r.heading != trace[0].heading for r in trace[first_trigger:]), "turned after"
    assert any(r.motors == (1.0, 0.0) for r in trace), "right motor was inhibited"


def test_agent_does_not_freeze() -> None:
    trace = run_episode(*build_scenario(), steps=200)
    assert all(r.motors != (0.0, 0.0) for r in trace), "some wheel is always driving"


def test_episode_is_reproducible() -> None:
    a = run_episode(*build_scenario(), steps=50)
    b = run_episode(*build_scenario(), steps=50)
    assert a == b
