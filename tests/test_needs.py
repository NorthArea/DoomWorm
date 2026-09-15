"""Stage 20: needs arbitration battery > call > clean (Plan §20.3)."""

from doomworm.brains import GradientFollower, NeedsArbiter, PlannerLayer
from doomworm.environments.sensors import IDEAL, SensorSuite
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
