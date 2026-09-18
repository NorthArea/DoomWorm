"""Stage 16: a place memory the worm does not have, as a layer outside the brain.

The reward decomposition says the same thing twice: no trained brain has ever
reached an exit, and the exit is the largest item on the board. The worm smells
the exit through walls and climbs that gradient straight into the wall between,
because nothing in it remembers where it has already pushed -- its neurons leak
over a few ticks and the contract has no map.

This layer supplies the missing *sense*, never a decision. It keeps a coarse
grid of the cells the agent's odometry has already visited and reports, in the
one currency a chemotaxis animal can read, where the unvisited ground lies:

    novelty_left / novelty_front / novelty_right

built exactly like the food gradient -- 1/distance to the nearest unvisited
cell, routed into the sector it falls in. The brain decides what to do with it,
and a brain with no route for those channels behaves exactly as before.

What this is not: it does not choose a waypoint, it does not steer, and it
never overrides the brain's output. Wrapping a brain in it is a *condition* of
the experiment and is reported as its own row in `docs/findings.md`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from wormlab.body import Drive
from wormlab.episode import BrainLike

NOVELTY = ("novelty_left", "novelty_front", "novelty_right")
FRONT_HALF_ANGLE = math.pi / 6  # the sector width every other gradient channel uses


class MemoryLayer:
    """A brain plus a memory of where it has been, expressed as a gradient.

    Args:
        inner: the brain that decides; it sees the extra channels and nothing else.
        cell: grid size in world units. One body length (1.0) by default.
        radius: how far the layer looks for unvisited ground, in world units.
        forget: ticks after which a visited cell counts as unvisited again; 0
            never forgets. A finite memory is the honest default -- the animal
            this models has no map, and an eternal one would make the layer a
            planner in disguise.
    """

    def __init__(
        self,
        inner: BrainLike,
        cell: float = 1.0,
        radius: float = 6.0,
        forget: int = 0,
        name: str | None = None,
    ) -> None:
        self.inner = inner
        self.cell = cell
        self.radius = radius
        self.forget = forget
        self.name = name or f"{getattr(inner, 'name', 'brain')}+memory"
        self.meta: dict[str, object] = {"layer": "memory", "cell": cell, "radius": radius}
        self.visited: dict[tuple[int, int], int] = {}
        self.tick = 0
        self.last_novelty: dict[str, float] = dict.fromkeys(NOVELTY, 0.0)

    # --- Brain interface ----------------------------------------------------------

    def reset(self) -> None:
        """Forget the episode, in the brain and in the memory."""
        self.inner.reset()
        self.visited.clear()
        self.tick = 0
        self.last_novelty = dict.fromkeys(NOVELTY, 0.0)

    def act(self, channels: Mapping[str, float]) -> tuple[float, float] | Drive:
        """Mark where we are, measure where we have not been, let the brain drive."""
        x, y = channels.get("odom_x", 0.0), channels.get("odom_y", 0.0)
        heading = channels.get("odom_heading", 0.0)
        self.visited[self._cell(x, y)] = self.tick
        self.last_novelty = self._novelty(x, y, heading)
        self.tick += 1
        return self.inner.act(dict(channels) | self.last_novelty)

    @property
    def fire(self) -> float:
        """The inner brain's trigger, untouched."""
        return float(getattr(self.inner, "fire", 0.0))

    @property
    def activity(self) -> dict[str, float] | None:
        """The inner brain's activity, for the debug screen."""
        return getattr(self.inner, "activity", None)

    # --- the memory ---------------------------------------------------------------

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return math.floor(x / self.cell), math.floor(y / self.cell)

    def _is_visited(self, cell: tuple[int, int]) -> bool:
        seen = self.visited.get(cell)
        if seen is None:
            return False
        return self.forget <= 0 or (self.tick - seen) < self.forget

    def _novelty(self, x: float, y: float, heading: float) -> dict[str, float]:
        """1/distance to the nearest unvisited cell, in its sector."""
        here = self._cell(x, y)
        reach = max(1, int(self.radius / self.cell))
        best: tuple[float, float, float] | None = None  # distance, dx, dy
        for i in range(-reach, reach + 1):
            for j in range(-reach, reach + 1):
                cell = (here[0] + i, here[1] + j)
                if self._is_visited(cell):
                    continue
                cx = (cell[0] + 0.5) * self.cell
                cy = (cell[1] + 0.5) * self.cell
                distance = math.hypot(cx - x, cy - y)
                if distance > self.radius or distance < 1e-9:
                    continue
                if best is None or distance < best[0]:
                    best = (distance, cx - x, cy - y)
        out = dict.fromkeys(NOVELTY, 0.0)
        if best is None:
            return out
        distance, dx, dy = best
        signal = 1.0 if distance <= 1.0 else 1.0 / distance
        bearing = math.atan2(dy, dx) - heading
        bearing = math.atan2(math.sin(bearing), math.cos(bearing))
        if abs(bearing) <= FRONT_HALF_ANGLE:
            out["novelty_front"] = signal
        elif bearing > 0.0:
            out["novelty_left"] = signal
        else:
            out["novelty_right"] = signal
        return out

    @property
    def explored(self) -> int:
        """How many cells the memory currently holds."""
        return sum(1 for c in self.visited if self._is_visited(c))
