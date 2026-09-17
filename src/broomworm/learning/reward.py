"""Per-tick reward (Plan §9). Fitness on stage 4 is the sum of this over an episode.

Movement is never rewarded directly. Rewarded events:

    ate food            +food
    reached target      +target
    inside danger       +damage per tick
    cleaned cells       +clean per cell (vacuum task); a cell is a fifth of a food item
    docked when low     +dock once per discharge cycle: docking with battery <= low_battery
                        pays only after the battery has been >= recharged since the last
                        paid docking (or since the start), so shuttling at the dock edge
                        earns nothing
    died (health 0)     +death once, episode ends
    collision           +collision per tick of contact, capped per episode
    new cell visited    +new_cell (cell_size x cell_size grid)
    starved             +starvation once, episode ends
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RewardConfig:
    """Reward scale. Single source of numbers for reward and fitness."""

    food: float = 10.0
    target: float = 10.0
    damage: float = -2.0
    death: float = -20.0
    clean: float = 0.5
    dock: float = 10.0
    low_battery: float = 0.3
    recharged: float = 0.9  # battery level that re-arms the docking bonus
    collision: float = -0.5
    max_collision_penalty: float = 20.0
    new_cell: float = 0.1
    cell_size: float = 1.0
    starvation: float = -20.0


@dataclass
class RewardTracker:
    """Accumulate reward over one episode."""

    config: RewardConfig = field(default_factory=RewardConfig)
    total: float = 0.0
    breakdown: dict[str, float] = field(
        default_factory=lambda: {
            "food": 0.0,
            "target": 0.0,
            "damage": 0.0,
            "death": 0.0,
            "clean": 0.0,
            "dock": 0.0,
            "collision": 0.0,
            "explore": 0.0,
            "starvation": 0.0,
        }
    )
    _visited: set[tuple[int, int]] = field(default_factory=set)
    _starved: bool = False
    _dead: bool = False
    _docked: bool = False
    _dock_armed: bool = True

    @property
    def cells_visited(self) -> int:
        """Number of distinct grid cells the agent has entered."""
        return len(self._visited)

    def step(
        self,
        *,
        x: float,
        y: float,
        ate: bool,
        collided: bool,
        starved: bool,
        reached: bool = False,
        damaged: bool = False,
        dead: bool = False,
        cleaned: int = 0,
        docked: bool = False,
        battery: float = 1.0,
    ) -> float:
        """Score one tick and return its reward."""
        reward = 0.0
        cfg = self.config

        if ate:
            reward += self._add("food", cfg.food)
        if reached:
            reward += self._add("target", cfg.target)
        if damaged:
            reward += self._add("damage", cfg.damage)
        if cleaned:
            reward += self._add("clean", cfg.clean * cleaned)
        if battery >= cfg.recharged:
            self._dock_armed = True
        if docked and not self._docked and battery <= cfg.low_battery and self._dock_armed:
            self._dock_armed = False
            reward += self._add("dock", cfg.dock)
        self._docked = docked
        if dead and not starved and not self._dead:
            self._dead = True
            reward += self._add("death", cfg.death)

        if collided:
            room = cfg.max_collision_penalty + self.breakdown["collision"]  # penalty left
            reward += self._add("collision", max(cfg.collision, -room))

        cell = (math.floor(x / cfg.cell_size), math.floor(y / cfg.cell_size))
        if cell not in self._visited:
            self._visited.add(cell)
            reward += self._add("explore", cfg.new_cell)

        if starved and not self._starved:
            self._starved = True
            reward += self._add("starvation", cfg.starvation)

        return reward

    def _add(self, key: str, value: float) -> float:
        self.breakdown[key] += value
        self.total += value
        return value
