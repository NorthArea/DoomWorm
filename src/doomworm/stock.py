"""Stage B11: a stock Doom scenario behind the same World contract (Plan §33).

Levels `vizdoom1`..`vizdoom6` are our own layouts written as a PWAD. This module
runs a map *nobody here designed* -- one of the scenarios shipped with ViZDoom --
and senses it with the same structured observations and no framebuffer:

    engine sector lines (blocking)  ->  the world's angled walls
    engine object list              ->  enemies
    game variables                  ->  pose, health, ammo

Everything else (rays, sectors, line of sight, `aim`, the trigger, the reward)
is the platform's existing code, so a brain trained on the mini-Doom levels runs
here unchanged. Plan §33 sets the bar honestly: survive, move, avoid walls,
react to enemies, occasionally attack -- not "win".

The layout is read once from a throwaway engine instance (the map is static),
then the world opens its own game on the same wad and seed, as every other
level does.

Optional dependency group ``doom`` (``uv sync --group doom``).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from doomworm.engine import MONSTERS, VizdoomWorld
from doomworm.wad import SCALE
from wormlab.environments.simple_2d import AgentState, Enemy, World

# name -> (scenario wad stem, map lump). Chosen for Plan §33's task list:
# survive, move, avoid walls, react to enemies, occasionally attack.
SCENARIOS: dict[str, tuple[str, str]] = {
    "stock_defend": ("defend_the_center", "map01"),  # ringed by monsters, must turn and shoot
    "stock_corridor": ("deadly_corridor", "map01"),  # move down a corridor under fire
    "stock_home": ("my_way_home", "map01"),  # rooms and corridors, no enemies: navigation only
}
STOCK_LEVELS = tuple(SCENARIOS)

# The classic game: Freedoom's free replacement for the original Doom data, fetched
# with `scripts/fetch_freedoom.py`. `e1m1` .. `e4m9` are the episode maps as the game
# ships them -- nobody's scenario, no exit we can reach (the classic exit is a switch),
# so these are combat and navigation levels for us.
IWAD = Path(__file__).resolve().parents[2] / "data/wads/freedoom1.wad"
CLASSIC_LEVELS = tuple(f"e{e}m{m}" for e in range(1, 5) for m in range(1, 10))


def is_classic_level(maps: str) -> bool:
    """True for `e1m1` .. `e4m9`."""
    return maps in CLASSIC_LEVELS


MARGIN = 2.0  # world units of padding between the map's bounding box and the world's border


def is_stock_level(maps: str) -> bool:
    """True for a bundled scenario or a classic episode map."""
    return maps in SCENARIOS or maps in CLASSIC_LEVELS


def scenario_wad(level: str) -> Path:
    """Where ViZDoom keeps the scenario that backs ``level``."""
    import vizdoom as vzd

    stem, _ = SCENARIOS[level]
    return Path(vzd.scenarios_path) / f"{stem}.wad"


def _configure(game: Any, level: str) -> None:
    """Point the engine at the right data: a bundled scenario, or the classic IWAD."""
    if level in SCENARIOS:
        _, map_lump = SCENARIOS[level]
        game.set_doom_scenario_path(str(scenario_wad(level)))
        game.set_doom_map(map_lump)
        return
    if not IWAD.exists():
        raise FileNotFoundError(
            f"{IWAD} is missing: run `uv run scripts/fetch_freedoom.py` for the classic maps"
        )
    game.set_doom_game_path(str(IWAD))
    game.set_doom_map(level.upper())


def _probe(level: str, seed: int) -> dict[str, Any]:
    """Open the level once and read its geometry, its monsters and the player."""
    import vizdoom as vzd

    game = vzd.DoomGame()
    _configure(game, level)
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_sectors_info_enabled(True)
    game.set_objects_info_enabled(True)
    game.set_available_game_variables(
        [
            vzd.GameVariable.POSITION_X,
            vzd.GameVariable.POSITION_Y,
            vzd.GameVariable.ANGLE,
            vzd.GameVariable.AMMO2,
        ]
    )
    game.set_mode(vzd.Mode.PLAYER)
    game.set_seed(seed)
    game.init()
    game.new_episode()
    state = game.get_state()
    lines = [
        (line.x1 / SCALE, line.y1 / SCALE, line.x2 / SCALE, line.y2 / SCALE)
        for sector in (state.sectors or [])
        for line in sector.lines
        if line.is_blocking
    ]
    monsters = [
        (o.position_x / SCALE, o.position_y / SCALE)
        for o in (state.objects or [])
        if o.name in MONSTERS
    ]
    player = (
        game.get_game_variable(vzd.GameVariable.POSITION_X) / SCALE,
        game.get_game_variable(vzd.GameVariable.POSITION_Y) / SCALE,
        math.radians(game.get_game_variable(vzd.GameVariable.ANGLE)),
    )
    ammo = int(game.get_game_variable(vzd.GameVariable.AMMO2))
    game.close()
    if not lines:
        raise ValueError(f"{level}: the engine reported no blocking lines")
    return {"lines": lines, "monsters": monsters, "player": player, "ammo": ammo}


def _layout(probe: dict[str, Any]) -> tuple[World, tuple[float, float]]:
    """The engine's map as a World: angled walls, monsters, the player's start."""
    lines: list[tuple[float, float, float, float]] = probe["lines"]
    xs = [v for line in lines for v in (line[0], line[2])]
    ys = [v for line in lines for v in (line[1], line[3])]
    offset = (MARGIN - min(xs), MARGIN - min(ys))
    ox, oy = offset
    px, py, heading = probe["player"]
    world = World(
        width=max(xs) - min(xs) + 2 * MARGIN,
        height=max(ys) - min(ys) + 2 * MARGIN,
        agent=AgentState(x=px + ox, y=py + oy, heading=heading),
        enemies=[Enemy(x + ox, y + oy, radius=0.5, health=1) for x, y in probe["monsters"]],
        fire_enabled=probe["ammo"] > 0,
        ammo=max(1, probe["ammo"]),
        hunger_rate=0.0,
    )
    world.segments = [(x1 + ox, y1 + oy, x2 + ox, y2 + oy) for x1, y1, x2, y2 in lines]
    return world, offset


def stock_world(seed: int, level: str = "stock_defend", tics: int = 1) -> VizdoomWorld:
    """A stock ViZDoom scenario as a World (stage B11).

    No exit and no generated map: the episode ends when the player dies or the
    loop's step cap is reached, and the reward is what the platform already
    pays for -- hits, kills, damage taken, collisions, new ground covered.
    """
    if not is_stock_level(level):
        raise ValueError(f"unknown level {level!r}, choose from {STOCK_LEVELS} or e1m1..e4m9")
    layout, offset = _layout(_probe(level, seed))
    return VizdoomWorld(
        layout,
        seed,
        level=level,
        tics=tics,
        wad=scenario_wad(level) if level in SCENARIOS else IWAD,
        map_lump=SCENARIOS[level][1] if level in SCENARIOS else level.upper(),
        as_iwad=level not in SCENARIOS,
        exit_ends=False,
        offset=offset,
    )
