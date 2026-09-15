"""The map + planner layer wrapped around any brain (Plan §3.2, stage 19).

    channels (odometry, rays) -> occupancy grid -> goal -> path -> waypoint
                                                    |
                              virtual gradient: target_left/front/right
                                                    v
                                              inner brain -> wheels

The inner brain never sees the map; it only receives a "smell" of the
next waypoint on the same channels a real target would use.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from doomworm.brains.base import Wheels
from doomworm.brains.needs import NeedsArbiter
from doomworm.environments.sensors import SensorConfig
from doomworm.episode import BrainLike
from doomworm.mapping import OccupancyGrid, nearest_unswept, next_waypoint, path_to

FRONT_HALF_ANGLE = math.pi / 6


def sector_channels(bearing: float, strength: float, prefix: str = "target") -> dict[str, float]:
    """Route one signal to the left / front / right channel by bearing (like the world does)."""
    left = front = right = 0.0
    if abs(bearing) <= FRONT_HALF_ANGLE:
        front = strength
    elif bearing > 0.0:
        left = strength
    else:
        right = strength
    return {f"{prefix}_left": left, f"{prefix}_front": front, f"{prefix}_right": right}


class PlannerLayer:
    """Wrap ``inner`` with a map, a goal source and a virtual gradient.

    Args:
        inner: any Brain.
        sensors: the sensor preset in use (ray angles and range are needed to map).
        width, height: world size (the map frame is the odometry frame).
        mode: ``"coverage"`` (go to the nearest unswept cell), ``"goal"``
            (go to :attr:`goal`), ``"needs"`` (stage 20: :class:`NeedsArbiter`
            picks dock / call / coverage), or ``"passthrough"`` (map only).
        strength: magnitude of the injected gradient (the worm's target
            neurons need about 0.5 to react).
        brush: sweep radius used to mark cells as done, in world units.
    """

    def __init__(
        self,
        inner: BrainLike,
        sensors: SensorConfig,
        width: float = 20.0,
        height: float = 20.0,
        mode: str = "coverage",
        strength: float = 0.6,
        brush: float = 1.0,
        cell: float = 0.5,
        replan_every: int = 5,
    ) -> None:
        if mode not in ("coverage", "goal", "needs", "passthrough"):
            raise ValueError("mode must be coverage, goal, needs or passthrough")
        self.inner = inner
        self.sensors = sensors
        self.width, self.height = width, height
        self.mode = mode
        self.strength = strength
        self.brush = brush
        self.cell = cell
        self.replan_every = replan_every
        self.name = f"planner[{getattr(inner, 'name', 'brain')}]"
        self.goal: tuple[float, float] | None = None
        self.grid = OccupancyGrid(width, height, cell)
        self.swept: set[tuple[int, int]] = set()
        self.path: list[tuple[int, int]] = []
        self.waypoint: tuple[float, float] | None = None
        self.pose = (0.0, 0.0, 0.0)
        self.tick = 0
        self.needs = NeedsArbiter()
        self.battery = 1.0

    def reset(self) -> None:
        """New episode: fresh map, fresh inner brain."""
        self.grid = OccupancyGrid(self.width, self.height, self.cell)
        self.swept = set()
        self.path = []
        self.waypoint = None
        self.tick = 0
        self.needs.reset()
        self.battery = 1.0
        self.inner.reset()

    # --- mapping ----------------------------------------------------------------

    def observe(self, channels: Mapping[str, float]) -> None:
        """Update pose, map and swept set from this tick's channels."""
        self.pose = (channels["odom_x"], channels["odom_y"], channels["odom_heading"])
        self.battery = channels.get("battery", 1.0)
        x, y, _ = self.pose
        readings = [channels.get(f"range_{i}", 0.0) for i in range(len(self.sensors.ray_angles))]
        self.grid.update(self.pose, self.sensors.ray_angles, readings, self.sensors.ray_range)
        self.grid.mark_free_around(x, y, self.brush)
        if channels.get("bumper_left", 0.0) or channels.get("bumper_right", 0.0):
            side = 0.0
            if not channels.get("bumper_left", 0.0):
                side = -0.5
            elif not channels.get("bumper_right", 0.0):
                side = 0.5
            heading = self.pose[2] + side
            bx, by = (
                x + math.cos(heading) * (self.brush + 0.3),
                y + math.sin(heading) * (self.brush + 0.3),
            )
            bi, bj = self.grid.to_cell(bx, by)
            self.grid.logodds[bi, bj] = self.grid.clamp  # a bump is certain
        i0, j0 = self.grid.to_cell(x - self.brush, y - self.brush)
        i1, j1 = self.grid.to_cell(x + self.brush, y + self.brush)
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                if math.dist(self.grid.to_world((i, j)), (x, y)) <= self.brush:
                    self.swept.add((i, j))

    def plan(self) -> None:
        """Choose the next waypoint for the current mode."""
        start = self.grid.to_cell(self.pose[0], self.pose[1])
        mode, goal = self.mode, self.goal
        if mode == "needs":
            _, need_goal = self.needs.decide(self.pose, self.battery)
            mode, goal = ("goal", need_goal) if need_goal is not None else ("coverage", None)
        if mode == "goal" and goal is not None:
            path = path_to(self.grid, start, self.grid.to_cell(*goal))
        elif mode == "coverage":
            path = nearest_unswept(self.grid, start, self.swept)
        else:
            path = None
        self.path = path or []
        self.waypoint = self.grid.to_world(next_waypoint(path)) if path else None

    def call(self, x: float, y: float) -> None:
        """Ask the robot to come to a point (needs mode)."""
        self.needs.request_call(x, y)

    def gradient(self) -> dict[str, float]:
        """Virtual target channels pointing at the waypoint (empty if none)."""
        if self.waypoint is None:
            return {}
        x, y, heading = self.pose
        bearing = math.atan2(self.waypoint[1] - y, self.waypoint[0] - x) - heading
        bearing = math.pi - (math.pi - bearing) % (2.0 * math.pi)
        return sector_channels(bearing, self.strength)

    # --- Brain interface ----------------------------------------------------------

    def act(self, channels: Mapping[str, float]) -> Wheels:
        """Map, plan every few ticks, inject the gradient, let the inner brain drive."""
        self.observe(channels)
        if self.tick % self.replan_every == 0 or self.waypoint is None:
            self.plan()
        self.tick += 1
        if self.parked():
            return 0.0, 0.0
        merged = dict(channels) | self.gradient()
        return self.inner.act(merged)

    def parked(self) -> bool:
        """Needs mode, charging and on the dock: hold the wheels, do not ask the brain."""
        if self.mode != "needs" or self.needs.state != "charge" or self.needs.dock is None:
            return False
        x, y, _ = self.pose
        return math.dist((x, y), self.needs.dock) < 0.4

    @property
    def activity(self) -> dict[str, float] | None:
        """Expose the inner brain's activity for the debug screen."""
        return getattr(self.inner, "activity", None)
