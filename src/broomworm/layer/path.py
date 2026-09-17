"""Path and coverage planning on an occupancy grid (Plan §3.2, stage 19). No learning."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable

from broomworm.layer.occupancy import OccupancyGrid

Cell = tuple[int, int]
NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _passable(grid: OccupancyGrid, clearance_cells: int) -> Callable[[Cell], bool]:
    def ok(cell: Cell) -> bool:
        i, j = cell
        if not grid.inside(i, j):
            return False
        for di in range(-clearance_cells, clearance_cells + 1):
            for dj in range(-clearance_cells, clearance_cells + 1):
                if grid.occupied(i + di, j + dj):
                    return False
        return True

    return ok


def bfs(
    grid: OccupancyGrid,
    start: Cell,
    goal_test: Callable[[Cell], bool],
    clearance_cells: int = 1,
) -> list[Cell] | None:
    """Shortest 4-connected path from ``start`` to the first cell satisfying ``goal_test``.

    Unknown cells are traversable (optimistic exploration); occupied cells and
    cells within ``clearance_cells`` of one are not. The start is always allowed.
    """
    ok = _passable(grid, clearance_cells)
    parents: dict[Cell, Cell | None] = {start: None}
    queue: deque[Cell] = deque([start])
    while queue:
        cell = queue.popleft()
        if goal_test(cell) and cell != start:
            path = [cell]
            while parents[path[-1]] is not None:
                nxt = parents[path[-1]]
                assert nxt is not None
                path.append(nxt)
            return path[::-1]
        for di, dj in NEIGHBOURS:
            nxt = (cell[0] + di, cell[1] + dj)
            if nxt not in parents and ok(nxt):
                parents[nxt] = cell
                queue.append(nxt)
    return None


def path_to(
    grid: OccupancyGrid, start: Cell, goal: Cell, clearance_cells: int = 1
) -> list[Cell] | None:
    """Path to one cell."""
    return bfs(grid, start, lambda c: c == goal, clearance_cells)


def nearest_unswept(
    grid: OccupancyGrid,
    start: Cell,
    swept: Iterable[Cell],
    clearance_cells: int = 1,
) -> list[Cell] | None:
    """Path to the closest cell (by path length) that is not occupied and not yet swept.

    Unknown cells count as unswept, which is what makes the robot explore.
    """
    done = set(swept)
    return bfs(grid, start, lambda c: c not in done and not grid.occupied(*c), clearance_cells)


def next_waypoint(path: list[Cell], lookahead: int = 2) -> Cell:
    """A few cells ahead on the path, so the follower does not zig-zag cell by cell."""
    return path[min(len(path) - 1, lookahead)]
