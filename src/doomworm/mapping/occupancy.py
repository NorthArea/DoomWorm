"""Occupancy grid from range readings and odometry (Plan §3.2, stage 19). No learning."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

UNKNOWN, FREE, OCCUPIED = 0, 1, 2


class OccupancyGrid:
    """Log-odds grid over the world rectangle; pose and rays come from the robot's own frame."""

    def __init__(
        self,
        width: float,
        height: float,
        cell: float = 0.5,
        hit: float = 0.9,
        miss: float = -0.6,
        clamp: float = 5.0,
    ) -> None:
        self.width, self.height, self.cell = width, height, cell
        self.cols, self.rows = round(width / cell), round(height / cell)
        self.logodds = np.zeros((self.cols, self.rows))
        self.hit, self.miss, self.clamp = hit, miss, clamp

    # --- coordinates ----------------------------------------------------------

    def to_cell(self, x: float, y: float) -> tuple[int, int]:
        """World point -> cell index (clamped inside the grid)."""
        i = min(self.cols - 1, max(0, int(x / self.cell)))
        j = min(self.rows - 1, max(0, int(y / self.cell)))
        return i, j

    def to_world(self, cell: tuple[int, int]) -> tuple[float, float]:
        """Cell index -> centre of the cell."""
        return (cell[0] + 0.5) * self.cell, (cell[1] + 0.5) * self.cell

    def inside(self, i: int, j: int) -> bool:
        """Whether the index is within the grid."""
        return 0 <= i < self.cols and 0 <= j < self.rows

    # --- state -------------------------------------------------------------------

    def state(self, i: int, j: int) -> int:
        """UNKNOWN / FREE / OCCUPIED by the sign of the log-odds."""
        v = self.logodds[i, j]
        if v > 0.5:
            return OCCUPIED
        if v < -0.5:
            return FREE
        return UNKNOWN

    def occupied(self, i: int, j: int) -> bool:
        """Occupied cell (out-of-grid counts as occupied)."""
        return not self.inside(i, j) or self.state(i, j) == OCCUPIED

    def known_fraction(self) -> float:
        """Share of cells that are no longer unknown."""
        return float(np.mean(np.abs(self.logodds) >= 0.5))

    def free_cells(self) -> list[tuple[int, int]]:
        """All FREE cells."""
        idx = np.argwhere(self.logodds < -0.5)
        return [(int(i), int(j)) for i, j in idx]

    # --- update --------------------------------------------------------------------

    def update(
        self,
        pose: tuple[float, float, float],
        ray_angles: Sequence[float],
        readings: Sequence[float],
        ray_range: float,
    ) -> None:
        """Integrate one scan: ``readings`` are proximities ``1 - d / range`` per ray."""
        x, y, heading = pose
        for angle, p in zip(ray_angles, readings, strict=True):
            hit = p > 0.0
            dist = (1.0 - p) * ray_range if hit else ray_range
            ang = heading + angle
            self._trace(x, y, ang, dist, hit)

    def _trace(self, x: float, y: float, ang: float, dist: float, hit: bool) -> None:
        step = self.cell / 2.0
        n = max(1, int(dist / step))
        last: tuple[int, int] | None = None
        for k in range(n + 1):
            d = min(dist, k * step)
            i, j = self.to_cell(x + math.cos(ang) * d, y + math.sin(ang) * d)
            if (i, j) == last:
                continue
            last = (i, j)
            if k == n and hit:
                self.logodds[i, j] = min(self.clamp, self.logodds[i, j] + self.hit)
            else:
                self.logodds[i, j] = max(-self.clamp, self.logodds[i, j] + self.miss)

    def mark_free_around(self, x: float, y: float, radius: float) -> None:
        """The robot's own footprint is free."""
        i0, j0 = self.to_cell(x - radius, y - radius)
        i1, j1 = self.to_cell(x + radius, y + radius)
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                cx, cy = self.to_world((i, j))
                if math.dist((cx, cy), (x, y)) <= radius:
                    self.logodds[i, j] = max(-self.clamp, self.logodds[i, j] + self.miss)

    def ascii(self, marks: dict[tuple[int, int], str] | None = None) -> str:
        """Top-down text view: ``#`` occupied, ``.`` free, space unknown."""
        marks = marks or {}
        lines = []
        for j in range(self.rows - 1, -1, -1):
            row = []
            for i in range(self.cols):
                if (i, j) in marks:
                    row.append(marks[(i, j)])
                else:
                    row.append({OCCUPIED: "#", FREE: ".", UNKNOWN: " "}[self.state(i, j)])
            lines.append("".join(row))
        return "\n".join(lines)
