"""Stage B11: a stock Doom scenario behind the same World contract (Plan §33).

Every test needs the optional ``doom`` group; without it they skip, like the
rest of the engine tests.
"""

from __future__ import annotations

import pytest

from doomworm.environments.worlds import build_world

pytest.importorskip("vizdoom")

from doomworm.environments.doom.stock import (
    SCENARIOS,
    STOCK_LEVELS,
    is_stock_level,
    stock_world,
)


def test_stock_level_names_are_known_and_separate_from_ours() -> None:
    assert is_stock_level("stock_defend")
    assert not is_stock_level("vizdoom4")
    assert not is_stock_level("doom4")
    assert set(STOCK_LEVELS) == set(SCENARIOS)


def test_the_engine_supplies_the_map_nobody_here_drew() -> None:
    """Geometry, monsters and the player's start all come from the scenario."""
    world = stock_world(1, "stock_defend")
    assert world.segments, "blocking sector lines become the world's angled walls"
    assert all(len(seg) == 4 for seg in world.segments)
    assert world.walls == [], "a stock map is not made of our boxes"
    assert world.obstacles == []
    assert world.enemies, "defend_the_center starts ringed by monsters"
    assert world.has_gun
    assert world.ammo_max == world.ammo, "the magazine the scenario grants"
    # everything sits inside the world's box, margin included
    assert 0.0 < world.agent.x < world.width
    assert 0.0 < world.agent.y < world.height
    for x1, y1, x2, y2 in world.segments:
        assert min(x1, x2) >= 0.0
        assert max(x1, x2) <= world.width
        assert min(y1, y2) >= 0.0
        assert max(y1, y2) <= world.height


def test_the_platform_senses_a_stock_map_with_its_own_code() -> None:
    """Rays, sectors and `aim` work on angled walls; no framebuffer is read."""
    world = stock_world(1, "stock_defend")
    channels = world.observe().as_channels()
    assert {"sensor_left", "sensor_front", "sensor_right", "aim", "ammo"} <= set(channels)
    assert all(0.0 <= channels[k] <= 1.0 for k in ("sensor_front", "aim", "ammo"))
    assert channels["ammo"] == pytest.approx(1.0), "full magazine at the start"
    assert world.ray_distance(0.0) > 0.0


def test_a_stock_episode_runs_and_never_ends_by_an_exit() -> None:
    world = stock_world(1, "stock_defend")
    start = (world.agent.x, world.agent.y)
    for i in range(20):
        obs = world.step(1.0, 0.7, fire=i % 5 == 0)
    assert not world.exited, "a stock scenario has no exit to reach"
    assert obs.as_channels()["ammo"] <= 1.0
    assert (world.agent.x, world.agent.y) != start, "the engine moved the player"


def test_build_world_knows_the_stock_levels() -> None:
    world = build_world(7, "stock_defend", "doom")
    assert world.segments
    with pytest.raises(ValueError, match="task 'doom'"):
        build_world(7, "stock_defend", "clean")
