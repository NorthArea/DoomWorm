"""Seeded world factories shared by every brain and the benchmark (Plan §3.3)."""

from __future__ import annotations

import math
from collections.abc import Callable

from wormlab.body import build_body
from wormlab.environments.maze import ApartmentConfig, MapConfig, apartment_world, random_world
from wormlab.environments.simple_2d import AgentState, Obstacle, World

N_FOOD = 2
BATTERY_DRAIN = 0.002  # 500 ticks from full to empty (Plan §8 range)


def build_world(
    seed: int,
    maps: str = "fixed",
    task: str = "food",
    dangers: int = 0,
    body: str = "differential",
) -> World:
    """Seeded world for any brain: ``fixed`` (stage 4 layout), ``random`` (§18), ``apartment``.

    ``doom1`` .. ``doom6`` are the mini-Doom levels (Plan §21-23, §27-32); their task
    is ``doom`` (exit, enemies, the gun) and ``dangers`` adds hazard zones on top.
    ``vizdoom1`` .. ``vizdoom6`` are the same layouts run by the Doom engine (Plan §24);
    ``stock_*`` are the scenarios shipped with ViZDoom, maps nobody here drew (Plan §33).
    ``body`` is the vehicle the world drives (stage 24): the platform's sensor preset
    names it, so the car moves on four mecanum wheels and the vacuum on two.
    """
    if maps.startswith("stock_") or _is_classic(maps):  # a map nobody here drew (Plan §33)
        from doomworm.stock import stock_world

        if task != "doom":
            raise ValueError("a doom level needs task 'doom'")
        return _with_body(stock_world(seed, maps), body)
    if maps.startswith("doom") or maps.startswith("vizdoom"):
        from doomworm.levels import doom_world

        if task != "doom":
            raise ValueError("a doom level needs task 'doom'")
        if maps.startswith("vizdoom"):  # the same level run by the Doom engine (optional group)
            from doomworm.engine import vizdoom_world

            return _with_body(vizdoom_world(seed, maps), body)
        world = doom_world(seed, maps)
        world.dangers = [world.spawn_danger() for _ in range(dangers)]
        return _with_body(world, body)
    if task == "doom":
        raise ValueError("task 'doom' needs a doom level (maps doom1..doom6)")
    if maps == "random":
        world = random_world(seed, MapConfig(n_food=N_FOOD, n_dangers=0))
        return _with_body(apply_task(world, task, dangers), body)
    if maps == "apartment":
        world = apartment_world(seed, ApartmentConfig(n_food=N_FOOD))
        return _with_body(apply_task(world, task, dangers), body)
    for prefix, factory in MAP_KINDS.items():
        if maps.startswith(prefix):
            return _with_body(factory(seed, maps, task, dangers), body)
    if maps != "fixed":
        known = ", ".join(["fixed", "random", "apartment", *(f"{p}..." for p in sorted(MAP_KINDS))])
        raise ValueError(f"unknown maps {maps!r}; this process knows: {known}")
    world = World(
        width=20.0,
        height=20.0,
        obstacles=[Obstacle(x=10.0, y=10.0, radius=1.5)],
        respawn_food=True,
        seed=seed,
    )
    rng = world.rng
    while True:
        x, y = rng.uniform(1.0, 19.0), rng.uniform(1.0, 19.0)
        heading = rng.uniform(-math.pi, math.pi)
        clear = all(math.dist((x, y), (o.x, o.y)) > o.radius + 1.0 for o in world.obstacles)
        if clear:
            break
    world.agent = AgentState(x=x, y=y, heading=heading)
    world.foods = [world.spawn_food() for _ in range(N_FOOD)]
    return _with_body(apply_task(world, task, dangers), body)


def _is_classic(maps: str) -> bool:
    """``e1m1`` .. ``e4m9``: the classic episode maps, from the Freedoom data."""
    from doomworm.stock import is_classic_level

    return is_classic_level(maps)


def _with_body(world: World, body: str) -> World:
    """Attach the platform's vehicle to a freshly built world."""
    world.body = build_body(body)
    return world


MapFactory = Callable[[int, str, str, int], World]

MAP_KINDS: dict[str, MapFactory] = {}
"""Map names a track owns, by prefix. See :func:`register_map`."""


def register_map(prefix: str, factory: MapFactory) -> None:
    """Let a track build worlds of its own from a name, e.g. ``room:<file>``.

    The third member of the same family as :func:`register_task` and
    :func:`wormlab.environments.sensors.register_preset`: a real room measured
    with a tape belongs to the track that owns the machine standing in it.
    """
    if prefix in MAP_KINDS:
        raise ValueError(f"map kind already registered: {prefix}")
    MAP_KINDS[prefix] = factory


TASKS: dict[str, Callable[[World], None]] = {}
"""Task overlays a track owns, by name. See :func:`register_task`."""


def register_task(name: str, overlay: Callable[[World], None]) -> None:
    """Let a track add its own task to the shared world (stage J2).

    The same arrangement as `register_preset`: a sensor preset describes a
    machine and a task describes what that machine is *for*, so both belong to
    the track rather than the platform. The vacuum's dirt, dock and battery live
    in the World because every track's simulator is the same simulator -- what
    the robot track owns is the decision to switch them on.

    Before this, splitting the tracks silently removed `clean` from the shared
    dispatcher and took the robot track's world-dependent commands with it.
    """
    if name in TASKS:
        raise ValueError(f"task already registered: {name}")
    TASKS[name] = overlay


def apply_task(world: World, task: str, dangers: int) -> World:
    """Task overlays: ``target`` (come to X), ``food`` as built, plus any a track registered."""
    if task == "target":
        world.foods = []
        world.respawn_target = True
        world.target = world.spawn_target()
    elif task in TASKS:
        TASKS[task](world)
    elif task != "food":
        known = ", ".join(sorted({"food", "target", *TASKS}))
        raise ValueError(f"unknown task {task!r}; this process knows: {known}")
    world.dangers = [world.spawn_danger() for _ in range(dangers)]
    return world
