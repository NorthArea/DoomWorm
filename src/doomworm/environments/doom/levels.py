"""Mini-Doom without Doom (Plan §21-23, §27-32): seeded levels on the 2D simulator.

The scenarios of Plan §27-32 by rising difficulty, every layout a deterministic
function of the seed so no brain ever sees one fixed map:

    doom1  empty room, find the exit                                   (§27, level 1)
    doom2  a partition with a doorway plus obstacles: navigate          (§28, level 2)
    doom3  level 2 plus one stationary enemy; no gun: avoid it          (§29, level 3 = §21)
    doom4  level 3 with the gun (50 rounds): aim roughly and shoot      (§30, level 4 = §22-23)
    doom5  the enemy walks toward the player                            (§31, level 5)
    doom6  several rooms (apartment), two walking enemies, the gun      (§32, level 6)

The exit is a :class:`~doomworm.environments.simple_2d.Target` that ends the
episode; an enemy is sensed on the ``danger_*`` channels and shot along the
front sector. Nothing here is Doom the game: that is the ViZDoom bridge.
"""

from __future__ import annotations

import math
import random

from doomworm.environments.maze import ApartmentConfig, apartment_world
from doomworm.environments.simple_2d import AgentState, Obstacle, Target, Wall, World

LEVELS = ("doom1", "doom2", "doom3", "doom4", "doom5", "doom6")
EXIT_RADIUS = 0.6
ENEMY_WALK = 0.1  # half the agent's top speed


def is_doom_level(maps: str) -> bool:
    """True for any Doom-task map: ours in the simulator or in the engine, or a stock one."""
    return (
        maps in LEVELS
        or maps in tuple(f"vizdoom{n}" for n in range(1, 7))
        or maps.startswith("stock_")  # stage B11: a scenario shipped with ViZDoom (Plan §33)
        or (len(maps) == 4 and maps[0] == "e" and maps[2] == "m" and maps[1::2].isdigit())
    )


def _spawn_exit(world: World, min_distance: float = 8.0) -> Target:
    """The exit: clear of solids and far from the start."""
    for _ in range(500):
        f = world.spawn_food(radius=EXIT_RADIUS, margin=1.2)
        if math.dist((f.x, f.y), (world.agent.x, world.agent.y)) >= min_distance:
            return Target(x=f.x, y=f.y, radius=EXIT_RADIUS)
    raise RuntimeError("could not place the exit")


def _room(seed: int, rng: random.Random) -> World:
    """An empty 20 x 20 room with the player at a random pose."""
    world = World(width=20.0, height=20.0, seed=seed, hunger_rate=0.0)
    x, y = rng.uniform(1.5, 18.5), rng.uniform(1.5, 18.5)
    world.agent = AgentState(x=x, y=y, heading=rng.uniform(-math.pi, math.pi))
    return world


def _corridor(seed: int, rng: random.Random) -> World:
    """Level 2 geometry: the player on one side of a partition, a doorway, obstacles."""
    world = World(width=20.0, height=20.0, seed=seed, hunger_rate=0.0)
    gap_y = rng.uniform(3.0, 17.0)
    gap = 3.0
    wx = rng.uniform(8.0, 12.0)
    world.walls = [
        Wall(x=wx, y=0.0, w=0.4, h=gap_y - gap / 2),
        Wall(x=wx, y=gap_y + gap / 2, w=0.4, h=20.0 - gap_y - gap / 2),
    ]
    obstacles: list[Obstacle] = []
    for _ in range(3):
        for _ in range(200):
            r = rng.uniform(0.6, 1.2)
            ox, oy = rng.uniform(r + 1.0, 19.0 - r), rng.uniform(r + 1.0, 19.0 - r)
            near_wall = abs(ox - wx) < r + 1.6
            near_other = any(
                math.dist((ox, oy), (o.x, o.y)) <= r + o.radius + 1.0 for o in obstacles
            )
            if not near_wall and not near_other:
                obstacles.append(Obstacle(x=ox, y=oy, radius=r))
                break
    world.obstacles = obstacles
    left_side = rng.random() < 0.5
    for _ in range(500):
        x = rng.uniform(1.5, wx - 1.5) if left_side else rng.uniform(wx + 2.0, 18.5)
        y = rng.uniform(1.5, 18.5)
        if world.clearance(x, y) > 1.0:
            break
    else:
        raise RuntimeError(f"could not place the player (seed {seed})")
    world.agent = AgentState(x=x, y=y, heading=rng.uniform(-math.pi, math.pi))
    # the exit is on the other side of the partition
    for _ in range(500):
        ex = rng.uniform(wx + 2.0, 18.5) if left_side else rng.uniform(1.5, wx - 1.5)
        ey = rng.uniform(1.5, 18.5)
        if world.clearance(ex, ey) > 1.2:
            world.target = Target(x=ex, y=ey, radius=EXIT_RADIUS)
            break
    else:
        raise RuntimeError(f"could not place the exit (seed {seed})")
    return world


def doom_world(seed: int, level: str) -> World:
    """Build the seeded mini-Doom world of ``level`` (one of :data:`LEVELS`)."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    n = int(level[4:])
    rng = random.Random(seed * 7919 + n)
    if n == 1:
        world = _room(seed, rng)
        world.target = _spawn_exit(world)
    elif n in (2, 3, 4, 5):
        world = _corridor(seed, rng)
    else:
        world = apartment_world(seed, ApartmentConfig(n_food=0))
        world.hunger_rate = 0.0
        world.foods = []
        world.target = _spawn_exit(world, min_distance=8.0)
    world.exit_ends = True
    world.respawn_target = False
    if n >= 3:
        speed = ENEMY_WALK if n >= 5 else 0.0
        count = 2 if n == 6 else 1
        world.enemies = [world.spawn_enemy(speed=speed) for _ in range(count)]
    if n >= 4:  # the gun arrives on level 4 with a Doom pistol's magazine
        world.has_gun = True
        world.ammo = world.ammo_max
    return world
