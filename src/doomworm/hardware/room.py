"""A real room as a simulator world (stage 22.2: the sim-vs-real check needs the same room).

A room file is JSON, in metres or in world units::

    {
      "units": "m",                 # "m" (converted with unit_m) or "u" (world units)
      "unit_m": 0.2,                # metres per world unit when units == "m"
      "width": 3.0, "height": 4.0,  # the floor, walls all around
      "walls": [[x, y, w, h], ...], # furniture and partitions as rectangles
      "obstacles": [[x, y, r], ...],# round things (chair legs, bins)
      "start": [x, y, heading_deg], # where the car is switched on
      "marker": [x, y]              # the dock marker, or null
    }

The world it builds carries the dirt map, the battery and the "dock" of the
clean task, so every brain and the needs layer run on it unchanged.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from doomworm.environments.simple_2d import AgentState, Dock, Obstacle, Wall, World

BATTERY_DRAIN = 0.002  # same as doomworm.worlds (imported there, not here: no cycle)


@dataclass(frozen=True)
class Room:
    """Geometry of one real room, in world units."""

    width: float
    height: float
    walls: list[tuple[float, float, float, float]] = field(default_factory=list)
    obstacles: list[tuple[float, float, float]] = field(default_factory=list)
    start: tuple[float, float, float] = (1.0, 1.0, 0.0)  # x, y, heading (radians)
    marker: tuple[float, float] | None = None
    name: str = "room"

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready (world units, radians)."""
        d = asdict(self)
        d["units"] = "u"
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Room:
        """Parse a room file dict; metres are converted with ``unit_m``."""
        units = data.get("units", "u")
        if units == "m":
            k = 1.0 / float(data.get("unit_m", 0.2))
        elif units == "u":
            k = 1.0
        else:
            raise ValueError("units must be 'm' or 'u'")
        sx, sy, heading = data.get("start", [1.0, 1.0, 0.0])
        if units == "m":  # a room file in metres gives the heading in degrees
            heading = math.radians(heading)
        marker = data.get("marker")
        return cls(
            width=float(data["width"]) * k,
            height=float(data["height"]) * k,
            walls=[(x * k, y * k, w * k, h * k) for x, y, w, h in data.get("walls", [])],
            obstacles=[(x * k, y * k, r * k) for x, y, r in data.get("obstacles", [])],
            start=(float(sx) * k, float(sy) * k, float(heading)),
            marker=None if marker is None else (float(marker[0]) * k, float(marker[1]) * k),
            name=str(data.get("name", "room")),
        )


def load_room(path: Path | str) -> Room:
    """Read a room file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "name" not in data:
        data["name"] = Path(path).stem
    return Room.from_dict(data)


def room_world(room: Room, seed: int = 0, task: str = "clean") -> World:
    """The room as a world: box walls, furniture, the car at its start, the marker as dock."""
    if task not in ("clean", "food", "target"):
        raise ValueError("task must be clean, food or target")
    x, y, heading = room.start
    world = World(
        width=room.width,
        height=room.height,
        agent=AgentState(x=x, y=y, heading=heading),
        walls=[Wall(*w) for w in room.walls],
        obstacles=[Obstacle(x=ox, y=oy, radius=r) for ox, oy, r in room.obstacles],
        seed=seed,
    )
    world.rooms = [(0.0, 0.0, room.width, room.height)]
    if task == "clean":
        world.hunger_rate = BATTERY_DRAIN
        if room.marker is not None:
            world.dock = Dock(x=room.marker[0], y=room.marker[1])
        world.init_dirt()
    elif task == "target":
        world.respawn_target = True
        world.target = world.spawn_target()
    return world
