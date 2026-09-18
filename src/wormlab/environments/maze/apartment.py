"""Apartment-like maps (Plan §20.2): a grid of rooms, doors in the partitions, furniture.

Everything is a deterministic function of the seed. The world gets ``rooms``
(bounding boxes) so trajectories can be scored by rooms visited.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from wormlab.environments.simple_2d import AgentState, Wall, World


@dataclass(frozen=True)
class ApartmentConfig:
    """Shape of the apartment family."""

    width: float = 20.0
    height: float = 20.0
    rooms: tuple[int, int] = (2, 2)  # columns, rows
    wall_thickness: float = 0.4
    door_width: float = 2.4
    door_margin: float = 1.2  # keep doors this far from wall ends
    furniture_per_room: int = 1
    furniture_size: tuple[float, float] = (1.0, 3.0)
    n_food: int = 2
    n_dangers: int = 0
    margin: float = 1.0
    respawn_food: bool = True
    tries: int = 500


def apartment_world(seed: int, config: ApartmentConfig | None = None) -> World:
    """Build a seeded apartment: rooms, doors, furniture, agent, food."""
    cfg = config or ApartmentConfig()
    rng = random.Random(seed)
    cols, rows = cfg.rooms
    rw, rh = cfg.width / cols, cfg.height / rows
    t = cfg.wall_thickness
    walls: list[Wall] = []

    # vertical partitions between column c and c+1, one door per room row
    for c in range(1, cols):
        x = c * rw - t / 2
        for r in range(rows):
            y0, y1 = r * rh, (r + 1) * rh
            door = rng.uniform(y0 + cfg.door_margin, y1 - cfg.door_margin - cfg.door_width)
            walls.append(Wall(x, y0, t, door - y0))
            walls.append(Wall(x, door + cfg.door_width, t, y1 - door - cfg.door_width))
    # horizontal partitions between row r and r+1, one door per room column
    for r in range(1, rows):
        y = r * rh - t / 2
        for c in range(cols):
            x0, x1 = c * rw, (c + 1) * rw
            door = rng.uniform(x0 + cfg.door_margin, x1 - cfg.door_margin - cfg.door_width)
            walls.append(Wall(x0, y, door - x0, t))
            walls.append(Wall(door + cfg.door_width, y, x1 - door - cfg.door_width, t))

    rooms = [(c * rw, r * rh, (c + 1) * rw, (r + 1) * rh) for r in range(rows) for c in range(cols)]

    world = World(
        width=cfg.width,
        height=cfg.height,
        agent=AgentState(x=rw / 2, y=rh / 2),
        walls=walls,
        respawn_food=cfg.respawn_food,
        seed=rng.randrange(2**31),
    )
    world.rooms = rooms

    # furniture: rectangles inside rooms, clear of walls/doors by the margin
    for x0, y0, x1, y1 in rooms:
        for _ in range(cfg.furniture_per_room):
            for _ in range(cfg.tries):
                fw, fh = rng.uniform(*cfg.furniture_size), rng.uniform(*cfg.furniture_size)
                fx = rng.uniform(x0 + cfg.margin + t, x1 - fw - cfg.margin - t)
                fy = rng.uniform(y0 + cfg.margin + t, y1 - fh - cfg.margin - t)
                piece = Wall(fx, fy, fw, fh)
                corners = [(fx, fy), (fx + fw, fy), (fx, fy + fh), (fx + fw, fy + fh)]
                if all(world.clearance(px, py) > cfg.margin for px, py in corners):
                    world.walls.append(piece)
                    break

    # agent: random clear pose in a random room
    for _ in range(cfg.tries):
        x0, y0, x1, y1 = rooms[rng.randrange(len(rooms))]
        x, y = (
            rng.uniform(x0 + cfg.margin, x1 - cfg.margin),
            rng.uniform(y0 + cfg.margin, y1 - cfg.margin),
        )
        if world.clearance(x, y) > cfg.margin:
            break
    else:
        raise RuntimeError(f"could not place the agent (seed {seed})")
    world.agent = AgentState(x=x, y=y, heading=rng.uniform(-math.pi, math.pi))
    world.foods = [world.spawn_food() for _ in range(cfg.n_food)]
    world.dangers = [world.spawn_danger() for _ in range(cfg.n_dangers)]
    return world


def rooms_visited(world: World, points: list[tuple[float, float]]) -> int:
    """Number of distinct rooms a trajectory passed through."""
    return len({idx for x, y in points if (idx := world.room_index(x, y)) is not None})
