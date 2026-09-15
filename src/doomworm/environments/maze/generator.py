"""Seeded random maps (Plan §18): random obstacles, random start, random food."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from doomworm.environments.simple_2d import AgentState, Obstacle, World


@dataclass(frozen=True)
class MapConfig:
    """Shape of the map family."""

    width: float = 20.0
    height: float = 20.0
    n_obstacles: int = 5
    obstacle_radius: tuple[float, float] = (0.8, 2.0)
    n_food: int = 2
    n_dangers: int = 0
    danger_radius: float = 1.0
    margin: float = 1.0  # keep obstacles and the agent this far from walls / each other
    respawn_food: bool = True
    tries: int = 500


def random_world(seed: int, config: MapConfig | None = None) -> World:
    """Build a world whose layout is a deterministic function of ``seed``."""
    cfg = config or MapConfig()
    rng = random.Random(seed)
    obstacles: list[Obstacle] = []
    for _ in range(cfg.n_obstacles):
        for _ in range(cfg.tries):
            r = rng.uniform(*cfg.obstacle_radius)
            x = rng.uniform(r + cfg.margin, cfg.width - r - cfg.margin)
            y = rng.uniform(r + cfg.margin, cfg.height - r - cfg.margin)
            if all(math.dist((x, y), (o.x, o.y)) > r + o.radius + cfg.margin for o in obstacles):
                obstacles.append(Obstacle(x=x, y=y, radius=r))
                break
        else:
            raise RuntimeError(f"could not place obstacle {len(obstacles) + 1} (seed {seed})")

    for _ in range(cfg.tries):
        x = rng.uniform(cfg.margin, cfg.width - cfg.margin)
        y = rng.uniform(cfg.margin, cfg.height - cfg.margin)
        if all(math.dist((x, y), (o.x, o.y)) > o.radius + cfg.margin for o in obstacles):
            break
    else:
        raise RuntimeError(f"could not place the agent (seed {seed})")
    agent = AgentState(x=x, y=y, heading=rng.uniform(-math.pi, math.pi))

    world = World(
        width=cfg.width,
        height=cfg.height,
        agent=agent,
        obstacles=obstacles,
        respawn_food=cfg.respawn_food,
        seed=rng.randrange(2**31),
    )
    world.foods = [world.spawn_food() for _ in range(cfg.n_food)]
    world.dangers = [world.spawn_danger(cfg.danger_radius) for _ in range(cfg.n_dangers)]
    return world
