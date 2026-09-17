"""Write a Doom map (PWAD) from a 2D world (Plan §24: a custom scenario, never a stock map).

One sector is the floor; box walls are the boundary, every :class:`Wall` and
:class:`Obstacle` becomes a solid pillar (a hole in the sector); the player
start, the exit marker (a green armour, harmless) and the monsters are THINGS.
No BSP nodes are written: the engine (ZDoom inside ViZDoom) builds them on load.

Doom map units: 1 world unit = :data:`SCALE` map units, so the player's
radius of 16 map units is the world's 0.5-unit body. Angles: degrees
counter-clockwise from +x, the same convention as the world's heading.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

from doomworm.environments.simple_2d import World

SCALE = 32.0  # map units per world unit
PLAYER_START = 1
EXIT_MARKER = 2018  # GreenArmor: visible, harmless
ZOMBIEMAN = 3004
SKILL_ALL = 0x0007
AMBUSH = 0x0008  # the monster waits until it sees the player
WALL_TEXTURE = b"STARTAN3"
FLOOR_FLAT = b"FLOOR0_1"
CEILING_FLAT = b"CEIL1_1"
INSET = 2  # map units pillars are kept away from the boundary (no colinear lines)

Point = tuple[int, int]


def _name(text: bytes | str) -> bytes:
    raw = text if isinstance(text, bytes) else text.encode()
    return raw.ljust(8, b"\0")[:8]


def _polygon_circle(x: float, y: float, r: float) -> list[Point]:
    """Octagon around a circle, counter-clockwise (the room lies on the right side)."""
    radius = r / math.cos(math.pi / 8)  # circumscribed: contains the circle
    return [
        (
            round(x + radius * math.cos(k * math.pi / 4)),
            round(y + radius * math.sin(k * math.pi / 4)),
        )
        for k in range(8)
    ]


def _polygon_rect(x0: float, y0: float, x1: float, y1: float) -> list[Point]:
    """Rectangle, counter-clockwise."""
    return [
        (round(x0), round(y0)),
        (round(x1), round(y0)),
        (round(x1), round(y1)),
        (round(x0), round(y1)),
    ]


def map_lumps(world: World, ambush: bool = True) -> list[tuple[str, bytes]]:
    """The lumps of MAP01 for ``world``: THINGS, LINEDEFS, SIDEDEFS, VERTEXES, SECTORS."""
    w, h = world.width * SCALE, world.height * SCALE
    polygons: list[list[Point]] = [
        [(0, 0), (0, round(h)), (round(w), round(h)), (round(w), 0)]
    ]  # clockwise
    lo, hi_x, hi_y = INSET, w - INSET, h - INSET
    for wall in world.walls:
        x0, y0 = max(lo, wall.x * SCALE), max(lo, wall.y * SCALE)
        x1, y1 = min(hi_x, wall.x1 * SCALE), min(hi_y, wall.y1 * SCALE)
        if x1 - x0 >= 2 and y1 - y0 >= 2:
            polygons.append(_polygon_rect(x0, y0, x1, y1))
    for o in world.obstacles:
        polygons.append(_polygon_circle(o.x * SCALE, o.y * SCALE, o.radius * SCALE))

    vertexes: list[Point] = []
    linedefs = b""
    sidedefs = b""
    for poly in polygons:
        base = len(vertexes)
        vertexes.extend(poly)
        n = len(poly)
        for i in range(n):
            side = len(sidedefs) // 30
            linedefs += struct.pack("<hhhhhhh", base + i, base + (i + 1) % n, 1, 0, 0, side, -1)
            sidedefs += struct.pack("<hh", 0, 0) + _name("-") + _name("-") + _name(WALL_TEXTURE)
            sidedefs += struct.pack("<h", 0)

    def thing(x: float, y: float, angle_deg: float, kind: int, flags: int) -> bytes:
        return struct.pack(
            "<hhhhh", round(x * SCALE), round(y * SCALE), round(angle_deg) % 360, kind, flags
        )

    a = world.agent
    things = thing(a.x, a.y, math.degrees(a.heading), PLAYER_START, SKILL_ALL)
    if world.target is not None:
        things += thing(world.target.x, world.target.y, 0, EXIT_MARKER, SKILL_ALL)
    for e in world.enemies:
        facing = math.degrees(math.atan2(a.y - e.y, a.x - e.x))
        things += thing(e.x, e.y, facing, ZOMBIEMAN, SKILL_ALL | (AMBUSH if ambush else 0))

    sectors = struct.pack("<hh", 0, 128) + _name(FLOOR_FLAT) + _name(CEILING_FLAT)
    sectors += struct.pack("<hhh", 160, 0, 0)
    vertex_blob = b"".join(struct.pack("<hh", x, y) for x, y in vertexes)
    return [
        ("MAP01", b""),
        ("THINGS", things),
        ("LINEDEFS", linedefs),
        ("SIDEDEFS", sidedefs),
        ("VERTEXES", vertex_blob),
        ("SEGS", b""),
        ("SSECTORS", b""),
        ("NODES", b""),
        ("SECTORS", sectors),
        ("REJECT", b""),
        ("BLOCKMAP", b""),
    ]


def write_wad(world: World, path: Path, ambush: bool = True) -> Path:
    """Write ``world`` as a one-map PWAD to ``path`` and return it."""
    lumps = map_lumps(world, ambush)
    data = b""
    directory = b""
    offset = 12
    for name, blob in lumps:
        directory += struct.pack("<ii", offset + len(data), len(blob)) + _name(name)
        data += blob
    header = b"PWAD" + struct.pack("<ii", len(lumps), 12 + len(data))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + data + directory)
    return path
