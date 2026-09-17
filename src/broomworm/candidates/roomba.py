"""A Roomba-style classical controller: the zero-learning floor of the A2 bake-off (Plan §20.4).

No map, no odometry, no learning. Reflexes on the vacuum sensor suite only:

    spiral     outward spiral on open floor (outer wheel full, inner wheel opening up)
    escape     bumper or cliff: back off, then turn away from the contact side,
               a little longer every time (corners)
    follow     after an escape, follow the wall on the right for a while
    dock       battery low: turn toward the dock beacon and drive; no beacon in
               range: keep wall-following until one appears
    charge     battery rising: sit still until full

A trained brain has to beat this row on the benchmark.
"""

from __future__ import annotations

from collections.abc import Mapping

from broomworm.candidates.base import Wheels


class RoombaBrain:
    """Hand-written state machine over bumper, cliff, wall IR, range and dock beacon."""

    def __init__(
        self,
        speed: float = 1.0,
        low: float = 0.4,
        full: float = 0.95,
        spiral_ticks: int = 150,
        follow_ticks: int = 120,
        name: str = "roomba",
    ) -> None:
        self.speed = speed
        self.low = low
        self.full = full
        self.spiral_ticks = spiral_ticks
        self.follow_ticks = follow_ticks
        self.name = name
        self.reset()

    def reset(self) -> None:
        """Forget every counter; start with a spiral."""
        self.state = "spiral"
        self.tick_in_state = 0
        self.bumps = 0
        self.escape: list[Wheels] = []
        self._last_battery: float | None = None

    def act(self, channels: Mapping[str, float]) -> Wheels:
        """Reflex priority: charging > escape > docking > follow / spiral."""
        battery = channels.get("battery", 1.0)
        charging = self._last_battery is not None and battery > self._last_battery
        self._last_battery = battery

        if self.state == "charge":
            if battery >= self.full:
                self._enter("spiral")
            elif charging:
                return 0.0, 0.0
            else:
                self._enter("dock")  # pushed off the dock, or never arrived

        contact = channels.get("bumper_left", 0.0) or channels.get("cliff_left", 0.0)
        contact_right = channels.get("bumper_right", 0.0) or channels.get("cliff_right", 0.0)
        if (contact or contact_right) and not self.escape:
            self._start_escape(turn_right=bool(contact) and not contact_right)
        if self.escape:
            wheels = self.escape.pop(0)
            if not self.escape and self.state not in ("dock", "charge"):
                self._enter("follow")
            return wheels

        if self.state == "dock" or battery <= self.low:
            if self.state != "dock":
                self._enter("dock")
            if charging:
                self._enter("charge")
                return 0.0, 0.0
            beacon = self._beacon(channels)
            if beacon is not None:
                return beacon
            return self._follow(channels)  # search along the walls

        if self.state == "follow" and self.tick_in_state >= self.follow_ticks:
            self._enter("spiral")
        self.tick_in_state += 1
        if self.state == "follow":
            return self._follow(channels)
        return self._spiral(channels)

    # --- behaviours --------------------------------------------------------------

    def _enter(self, state: str) -> None:
        self.state = state
        self.tick_in_state = 0

    def _start_escape(self, turn_right: bool) -> None:
        self.bumps += 1
        turn = 6 + 3 * (self.bumps % 4)
        spin: Wheels = (self.speed, -self.speed) if turn_right else (-self.speed, self.speed)
        self.escape = [(-0.6, -0.6)] * 3 + [spin] * turn

    def _spiral(self, channels: Mapping[str, float]) -> Wheels:
        if self._wall_ahead(channels):
            return self._veer(channels)
        inner = 0.2 + 0.8 * min(1.0, self.tick_in_state / self.spiral_ticks)
        return self.speed, self.speed * inner

    def _follow(self, channels: Mapping[str, float]) -> Wheels:
        if self._wall_ahead(channels):
            return -self.speed, self.speed  # wall on the right: turn left in place
        wall = channels.get("wall_right", 0.0)
        if wall < 0.3:
            return self.speed, self.speed * 0.6  # lost it: bend right
        if wall > 0.7:
            return self.speed * 0.6, self.speed  # too close: bend left
        return self.speed, self.speed

    def _beacon(self, channels: Mapping[str, float]) -> Wheels | None:
        left = channels.get("dock_left", 0.0)
        front = channels.get("dock_front", 0.0)
        right = channels.get("dock_right", 0.0)
        if max(left, front, right) <= 0.0:
            return None
        if self._wall_ahead(channels):
            return self._veer(channels)
        if front >= max(left, right):
            return self.speed, self.speed
        if left > right:
            return self.speed * 0.3, self.speed
        return self.speed, self.speed * 0.3

    @staticmethod
    def _wall_ahead(channels: Mapping[str, float]) -> bool:
        return channels.get("sensor_front", 0.0) > 0.7

    def _veer(self, channels: Mapping[str, float]) -> Wheels:
        left_closer = channels.get("sensor_left", 0.0) > channels.get("sensor_right", 0.0)
        return (self.speed, -self.speed) if left_closer else (-self.speed, self.speed)
