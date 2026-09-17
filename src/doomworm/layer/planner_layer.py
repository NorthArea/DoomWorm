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

from doomworm.candidates.base import Wheels
from doomworm.environments.sensors import SensorConfig
from doomworm.episode import BrainLike
from doomworm.layer.follower import GradientFollower
from doomworm.layer.needs import NeedsArbiter
from doomworm.layer.occupancy import OccupancyGrid
from doomworm.layer.path import nearest_unswept, next_waypoint, path_to

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
        footprint: body radius (the world's ``agent_radius``) marked free on the map
            every tick; larger than the body, and walls the robot drives up to are
            erased from the map.
        beacon_homing: dock beacon strength (1/distance) from which the beacon bearing
            replaces the planned path on the final approach; 0.5 = within 2 units.
            Further out the beacon may point through a wall.
        cost_per_cell: battery the inner brain spends per path cell (measured on the
            scripted driver: ~4 ticks per 0.5 cell at drain 0.002), used to decide
            when to leave for the dock.
        dock_autopilot: in needs mode, drive the trip to the dock with the layer's
            own gradient follower instead of the inner brain (stage 21.8): return
            to dock is a safety routine of the platform, the brain only cleans.
        bumper_reflex: on bumper or cliff contact the layer backs off for 3 ticks
            and spins away from the contact side, longer with every repeated bump
            (stage 21.9); the brain is not consulted during the manoeuvre.
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
        footprint: float = 0.5,
        cell: float = 0.5,
        replan_every: int = 5,
        beacon_homing: float = 0.5,
        cost_per_cell: float = 0.008,
        dock_autopilot: bool = True,
        bumper_reflex: bool = True,
        marker_search: bool = True,
        search_radius: float = 3.0,
    ) -> None:
        if mode not in ("coverage", "goal", "needs", "passthrough"):
            raise ValueError("mode must be coverage, goal, needs or passthrough")
        self.inner = inner
        self.sensors = sensors
        self.width, self.height = width, height
        self.mode = mode
        self.strength = strength
        self.brush = brush
        self.footprint = footprint
        self.cell = cell
        self.replan_every = replan_every
        self.beacon_homing = beacon_homing
        self.cost_per_cell = cost_per_cell
        self.dock_autopilot = dock_autopilot
        self.autopilot = GradientFollower(name="autopilot")
        self.bumper_reflex = bumper_reflex
        # Camera marker (stage 22): the dock is a landmark the camera sees within its field
        # of view. Any sighting is line of sight, so it is trusted at any strength, it
        # re-anchors the dock estimate, and near the estimated dock without a sighting the
        # layer spins to look for it (dead reckoning without encoders drifts by units).
        self.camera = sensors.beacon_fov is not None
        self.marker_search = marker_search
        self.search_radius = search_radius
        self._search: list[Wheels] = []
        self._search_h0: float | None = None
        self.escape: list[Wheels] = []
        self.bumps = 0
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
        self._last_battery: float | None = None
        self.charging = False
        self.beacon = (0.0, 0.0, 0.0)

    def reset(self) -> None:
        """New episode: fresh map, fresh inner brain."""
        self.grid = OccupancyGrid(self.width, self.height, self.cell)
        self.swept = set()
        self.path = []
        self.waypoint = None
        self.tick = 0
        self.needs.reset()
        self.battery = 1.0
        self._last_battery = None
        self.charging = False
        self.beacon = (0.0, 0.0, 0.0)
        self.autopilot.reset()
        self.escape = []
        self.bumps = 0
        self._search = []
        self._search_h0 = None
        self.inner.reset()

    # --- mapping ----------------------------------------------------------------

    def observe(self, channels: Mapping[str, float]) -> None:
        """Update pose, map and swept set from this tick's channels."""
        self.pose = (channels["odom_x"], channels["odom_y"], channels["odom_heading"])
        self.battery = channels.get("battery", 1.0)
        # A rising battery means the robot sits on the dock (a real robot has a charging
        # flag). Re-anchor the dock estimate there: the odometry frame drifts between visits.
        self.charging = self._last_battery is not None and self.battery > self._last_battery
        self._last_battery = self.battery
        self.beacon = (
            channels.get("dock_left", 0.0),
            channels.get("dock_front", 0.0),
            channels.get("dock_right", 0.0),
        )
        x, y, heading = self.pose
        if self.mode == "needs" and self.charging:
            self.needs.dock = (x, y)
        elif self.mode == "needs" and self.camera and max(self.beacon) > 0.0:
            self.needs.dock = self._marker_position(x, y, heading)
        readings = [channels.get(f"range_{i}", 0.0) for i in range(len(self.sensors.ray_angles))]
        self.grid.update(self.pose, self.sensors.ray_angles, readings, self.sensors.ray_range)
        self.grid.mark_free_around(x, y, self.footprint)
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

    def _marker_position(self, x: float, y: float, heading: float) -> tuple[float, float]:
        """Where the marker is, from its sector and apparent size (strength = 1 / distance)."""
        left, front, right = self.beacon
        strength = max(self.beacon)
        distance = 1.0 / strength if strength < 1.0 else 1.0
        fov = self.sensors.beacon_fov or 0.0
        bearing = (
            0.0
            if front >= max(left, right)
            else (2.0 * fov / 3.0 if left > right else -2.0 * fov / 3.0)
        )
        return (
            x + distance * math.cos(heading + bearing),
            y + distance * math.sin(heading + bearing),
        )

    def plan(self) -> None:
        """Choose the next waypoint for the current mode."""
        start = self.grid.to_cell(self.pose[0], self.pose[1])
        mode, goal = self.mode, self.goal
        if mode == "needs":
            _, need_goal = self.needs.decide(self.pose, self.battery, self.trip_cost(start))
            mode, goal = ("goal", need_goal) if need_goal is not None else ("coverage", None)
        if mode == "goal" and goal is not None:
            path = path_to(self.grid, start, self.grid.to_cell(*goal))
        elif mode == "coverage":
            path = nearest_unswept(self.grid, start, self.swept)
        else:
            path = None
        self.path = path or []
        self.waypoint = self.grid.to_world(next_waypoint(path)) if path else None
        if self.waypoint is None and mode == "goal" and goal is not None:
            self.waypoint = goal  # no path through the known map: head straight for it

    def trip_cost(self, start: tuple[int, int]) -> float:
        """Battery the way home is expected to cost: path length on the map, else beeline."""
        if self.needs.dock is None:
            return 0.0
        dock = self.grid.to_cell(*self.needs.dock)
        path = path_to(self.grid, start, dock)
        cells = len(path) if path else abs(dock[0] - start[0]) + abs(dock[1] - start[1])
        return self.cost_per_cell * cells

    def call(self, x: float, y: float) -> None:
        """Ask the robot to come to a point (needs mode)."""
        self.needs.request_call(x, y)

    def gradient(self) -> dict[str, float]:
        """Virtual target channels pointing at the waypoint (empty if none).

        Heading for the dock with its beacon in range, the beacon bearing wins over
        the planned waypoint: the beacon is measured, the waypoint is dead reckoning.
        """
        if self.homing():
            left, front, right = self.beacon
            return {
                "target_left": self.strength * float(left > 0.0),
                "target_front": self.strength * float(front > 0.0),
                "target_right": self.strength * float(right > 0.0),
            }
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
        if self.autopiloting():
            if self.searching():
                if self.bumper_reflex:
                    self._reflex(channels)
                    if self.escape:
                        self._search, self._search_h0 = [], None
                        return self.escape.pop(0)
                return self._search_step()
            self._search, self._search_h0 = [], None
            return self.autopilot.act(merged)  # the autopilot has its own escape
        if self.bumper_reflex:
            self._reflex(channels)
            if self.escape:
                return self.escape.pop(0)
        return self.inner.act(merged)

    def _reflex(self, channels: Mapping[str, float]) -> None:
        """Queue an escape manoeuvre on a fresh contact; nothing while one is running."""
        if self.escape:
            return
        left = channels.get("bumper_left", 0.0) or channels.get("cliff_left", 0.0)
        right = channels.get("bumper_right", 0.0) or channels.get("cliff_right", 0.0)
        if not (left or right):
            self.bumps = max(0, self.bumps - 1) if self.tick % 50 == 0 else self.bumps
            return
        self.bumps += 1
        turn = 4 + 2 * (self.bumps % 4)
        spin: Wheels = (1.0, -1.0) if left and not right else (-1.0, 1.0)
        self.escape = [(-0.6, -0.6)] * 3 + [spin] * turn

    def autopiloting(self) -> bool:
        """Needs mode, heading for the dock, autopilot on: the layer drives."""
        return self.dock_autopilot and self.mode == "needs" and self.needs.state == "charge"

    def homing(self) -> bool:
        """Needs mode, heading for the dock, beacon strong enough to trust its bearing.

        A camera marker is line of sight: any sighting is trusted. The IR beacon of
        the vacuum preset reaches through walls, so it needs ``beacon_homing``.
        """
        threshold = 0.0 if self.camera else self.beacon_homing
        return (
            self.mode == "needs"
            and self.needs.state == "charge"
            and max(self.beacon) > threshold
            and (max(self.beacon) >= self.beacon_homing or self.camera)
        )

    def searching(self) -> bool:
        """Heading for a camera marker, none in sight, dead reckoning says it is near."""
        if not (self.camera and self.marker_search and self.needs.state == "charge"):
            return False
        if max(self.beacon) > 0.0 or self.needs.dock is None:
            return False
        x, y, _ = self.pose
        near = math.dist((x, y), self.needs.dock) <= self.search_radius
        return near or self.waypoint is None or self._search_h0 is not None

    def _search_step(self) -> Wheels:
        """Spin a full turn on the odometry heading, then move on a little, repeat."""
        heading = self.pose[2]
        if self._search_h0 is None:
            self._search_h0 = heading
            self._search = []
        if self._search:
            return self._search.pop(0)
        turned = abs(heading - self._search_h0)
        if turned < math.tau:
            return 1.0, -1.0  # spin in place: the camera sweeps the room
        # a full turn without a sighting: change the vantage point and look again
        self._search = [(1.0, 1.0)] * 8
        self._search_h0 = heading
        return self._search.pop(0)

    def parked(self) -> bool:
        """Needs mode, heading for the dock and charging: hold the wheels, skip the brain."""
        return self.mode == "needs" and self.needs.state == "charge" and self.charging

    @property
    def activity(self) -> dict[str, float] | None:
        """Expose the inner brain's activity for the debug screen."""
        return getattr(self.inner, "activity", None)

    @property
    def fire(self) -> float:
        """The inner brain's trigger (track B); the layer never shoots on its own."""
        return float(getattr(self.inner, "fire", 0.0))
