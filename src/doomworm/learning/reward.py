"""Per-tick reward (Plan §9). Fitness on stage 4 is the sum of this over an episode.

Movement is never rewarded directly. Rewarded events:

    ate food            +food
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
        default_factory=lambda: {"food": 0.0, "collision": 0.0, "explore": 0.0, "starvation": 0.0}
    )
    _visited: set[tuple[int, int]] = field(default_factory=set)
    _starved: bool = False

    @property
    def cells_visited(self) -> int:
        """Number of distinct grid cells the agent has entered."""
        return len(self._visited)

    def step(self, *, x: float, y: float, ate: bool, collided: bool, starved: bool) -> float:
        """Score one tick and return its reward."""
        reward = 0.0
        cfg = self.config

        if ate:
            reward += self._add("food", cfg.food)

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
