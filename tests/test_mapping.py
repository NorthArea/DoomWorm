"""Stage 19: occupancy grid, planner, planner layer (Plan §3.2, §20.3)."""

import math
from collections.abc import Mapping
from itertools import pairwise

import pytest

from doomworm.brains import GradientFollower, PlannerLayer, ScriptedBrain, sector_channels
from doomworm.environments.sensors import IDEAL, VACUUM, SensorSuite
from doomworm.environments.simple_2d import AgentState, Target, Wall, World
from doomworm.episode import run_brain_episode
from doomworm.mapping import FREE, OCCUPIED, UNKNOWN, OccupancyGrid, nearest_unswept, path_to
from doomworm.worlds import build_world


def test_grid_marks_free_and_occupied() -> None:
    g = OccupancyGrid(10.0, 10.0, cell=0.5)
    assert g.state(*g.to_cell(5.0, 5.0)) == UNKNOWN
    # robot at (2, 5) facing +x, wall hit at distance 3 with range 4 -> proximity 0.25
    for _ in range(3):
        g.update((2.0, 5.0, 0.0), [0.0], [0.25], 4.0)
    assert g.state(*g.to_cell(3.5, 5.0)) == FREE
    assert g.state(*g.to_cell(5.0, 5.0)) == OCCUPIED
    assert g.state(*g.to_cell(5.0, 8.0)) == UNKNOWN
    assert 0.0 < g.known_fraction() < 0.2
    assert "#" in g.ascii()


def test_no_hit_marks_full_range_free() -> None:
    g = OccupancyGrid(10.0, 10.0, cell=0.5)
    g.update((1.0, 5.0, 0.0), [0.0], [0.0], 4.0)
    assert g.state(*g.to_cell(4.75, 5.0)) == FREE
    assert g.state(*g.to_cell(6.0, 5.0)) == UNKNOWN


def test_bfs_goes_around_a_wall() -> None:
    g = OccupancyGrid(10.0, 10.0, cell=1.0)
    for j in range(0, 7):  # wall x=5 from y=0..7, gap at the top
        g.logodds[5, j] = 5.0
    start, goal = g.to_cell(2.5, 2.5), g.to_cell(7.5, 2.5)
    path = path_to(g, start, goal, clearance_cells=0)
    assert path is not None
    assert path[0] == start
    assert path[-1] == goal
    assert all(not g.occupied(*c) for c in path)
    assert max(j for _, j in path) >= 7, "went over the wall's top end"
    g.logodds[5, :] = 5.0  # seal it
    assert path_to(g, start, goal, clearance_cells=0) is None


def test_nearest_unswept_prefers_close_cells_and_explores_unknown() -> None:
    g = OccupancyGrid(6.0, 6.0, cell=1.0)
    start = (0, 0)
    swept = {(0, 0), (1, 0), (0, 1)}
    path = nearest_unswept(g, start, swept, clearance_cells=0)
    assert path is not None
    assert path[-1] in {(2, 0), (1, 1), (0, 2)}
    assert path[-1] not in swept


def test_sector_channels() -> None:
    assert sector_channels(0.0, 0.5) == {
        "target_left": 0.0,
        "target_front": 0.5,
        "target_right": 0.0,
    }
    assert sector_channels(1.0, 0.5)["target_left"] == 0.5
    assert sector_channels(-1.0, 0.5)["target_right"] == 0.5


def test_layer_injects_gradient_and_maps() -> None:
    seen: list[dict[str, float]] = []

    def spy(ch: Mapping[str, float]) -> tuple[float, float]:
        seen.append(dict(ch))
        return 1.0, 1.0

    world = World(agent=AgentState(x=3.0, y=10.0), walls=[Wall(12.0, 0.0, 1.0, 20.0)])
    layer = PlannerLayer(ScriptedBrain(spy), IDEAL, mode="coverage")
    run_brain_episode(world, layer, 30, sensors=SensorSuite(IDEAL))
    assert layer.grid.known_fraction() > 0.02
    assert layer.swept
    assert any(
        ch.get("target_front", 0) or ch.get("target_left", 0) or ch.get("target_right", 0)
        for ch in seen
    )
    assert layer.name == "planner[scripted]"


def test_goal_mode_reaches_another_room_with_the_follower() -> None:
    """Acceptance: 'come to X' in a neighbouring room on an unseen apartment."""
    world = build_world(3004, "apartment", "target")
    start_room = world.room_index(world.agent.x, world.agent.y)
    assert start_room is not None
    goal_room = (start_room + 1) % len(world.rooms)
    x0, y0, x1, y1 = world.rooms[goal_room]
    world.target = Target(x=(x0 + x1) / 2, y=(y0 + y1) / 2)
    layer = PlannerLayer(GradientFollower(), IDEAL, mode="goal", replan_every=3)
    layer.goal = (world.target.x, world.target.y)
    trace = run_brain_episode(world, layer, 600, sensors=SensorSuite(IDEAL))
    reached = any(r.reached for r in trace)
    rooms = {world.room_index(r.x, r.y) for r in trace}
    assert reached or (goal_room in rooms and goal_room != start_room), (
        f"start room {start_room}, goal room {goal_room}, visited {rooms}"
    )


def test_coverage_mode_explores_an_apartment() -> None:
    world = build_world(3001, "apartment", "clean")
    layer = PlannerLayer(GradientFollower(), VACUUM, mode="coverage")
    trace = run_brain_episode(world, layer, 400, sensors=SensorSuite(VACUUM, seed=1))
    assert layer.grid.known_fraction() > 0.15
    assert world.coverage > 0.1
    distance = sum(math.dist((a.x, a.y), (b.x, b.y)) for a, b in pairwise(trace))
    assert distance > 20.0


def test_layer_rejects_bad_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        PlannerLayer(GradientFollower(), IDEAL, mode="nope")
