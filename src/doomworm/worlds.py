"""Seeded world factories shared by every brain and the benchmark (Plan §3.3)."""

from __future__ import annotations

import math

from doomworm.environments.maze import ApartmentConfig, MapConfig, apartment_world, random_world
from doomworm.environments.simple_2d import AgentState, Dock, Obstacle, World

N_FOOD = 2
BATTERY_DRAIN = 0.002  # 500 ticks from full to empty (Plan §8 range)


def build_world(seed: int, maps: str = "fixed", task: str = "food", dangers: int = 0) -> World:
    """Seeded world for any brain: ``fixed`` (stage 4 layout), ``random`` (§18), ``apartment``."""
    if maps == "random":
        return apply_task(random_world(seed, MapConfig(n_food=N_FOOD, n_dangers=0)), task, dangers)
    if maps == "apartment":
        return apply_task(apartment_world(seed, ApartmentConfig(n_food=N_FOOD)), task, dangers)
    if maps.startswith("room:"):  # stage 22: a real room from a file (hardware/room.py)
        from doomworm.hardware.room import load_room, room_world

        world = room_world(load_room(maps[len("room:") :]), seed, task)
        world.dangers = [world.spawn_danger() for _ in range(dangers)]
        return world
    if maps != "fixed":
        raise ValueError("maps must be 'fixed', 'random', 'apartment' or 'room:<file>'")
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
    return apply_task(world, task, dangers)


def apply_task(world: World, task: str, dangers: int) -> World:
    """Task overlays: ``target`` (come to X), ``clean`` (vacuum: dirt, dock, battery)."""
    if task == "target":
        world.foods = []
        world.respawn_target = True
        world.target = world.spawn_target()
    elif task == "clean":
        world.foods = []
        world.hunger_rate = BATTERY_DRAIN
        spot = world.spawn_food(radius=0.6, margin=1.2)
        world.dock = Dock(x=spot.x, y=spot.y)
        world.agent = AgentState(x=spot.x, y=spot.y, heading=world.agent.heading)
        world.init_dirt()
    elif task != "food":
        raise ValueError("task must be 'food', 'target' or 'clean'")
    world.dangers = [world.spawn_danger() for _ in range(dangers)]
    return world
