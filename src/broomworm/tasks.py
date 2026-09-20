"""The robot track's task: clean the floor, on a battery, from a dock (stage J2).

The shared World already carries the dirt map, the dock and the battery -- one
simulator serves both tracks. What this track owns is switching them on, and it
does that the way it registers its sensor presets: by declaring itself to the
platform at import.

Importing `broomworm` is enough; `broomworm.cli` and `broomworm.presets` both
pull this in, so any command of this track has the task available.
"""

from __future__ import annotations

from wormlab.environments.simple_2d import AgentState, Dock, World
from wormlab.environments.worlds import BATTERY_DRAIN, register_task


def clean(world: World) -> None:
    """Dirt everywhere, a dock where the robot starts, and a battery that runs down."""
    world.foods = []
    world.hunger_rate = BATTERY_DRAIN
    spot = world.spawn_food(radius=0.6, margin=1.2)
    world.dock = Dock(x=spot.x, y=spot.y)
    world.agent = AgentState(x=spot.x, y=spot.y, heading=world.agent.heading)
    world.init_dirt()


register_task("clean", clean)
