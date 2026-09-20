"""The robot track's worlds: the `clean` task, and rooms measured with a tape.

The shared World already carries the dirt map, the dock and the battery -- one
simulator serves both tracks. What this track owns is switching them on, and it
does that the way it registers its sensor presets: by declaring itself to the
platform at import.

A `room:<file>` map is the same idea: a real room, described in metres in
`data/rooms/*.json`, is this track's world and not the platform's, so it is
registered here rather than branched on inside `build_world` (stage J4).

Importing `broomworm` is enough; `broomworm.cli` and `broomworm.presets` both
pull this in, so any command of this track has both available.
"""

from __future__ import annotations

from wormlab.environments.simple_2d import AgentState, Dock, World
from wormlab.environments.worlds import BATTERY_DRAIN, register_map, register_task


def clean(world: World) -> None:
    """Dirt everywhere, a dock where the robot starts, and a battery that runs down."""
    world.foods = []
    world.hunger_rate = BATTERY_DRAIN
    spot = world.spawn_food(radius=0.6, margin=1.2)
    world.dock = Dock(x=spot.x, y=spot.y)
    world.agent = AgentState(x=spot.x, y=spot.y, heading=world.agent.heading)
    world.init_dirt()


def room_map(seed: int, maps: str, task: str, dangers: int) -> World:
    """``room:<file>``: the world of a room file, with the task and hazards applied."""
    from broomworm.hardware.room import load_room, room_world

    world = room_world(load_room(maps[len("room:") :]), seed, task)
    world.dangers = [world.spawn_danger() for _ in range(dangers)]
    return world


register_task("clean", clean)
register_map("room:", room_map)
