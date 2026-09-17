"""Tiny hand-written brains: platform test drivers, not candidates."""

from __future__ import annotations

from collections.abc import Mapping

from broomworm.candidates.base import Wheels


class GradientFollower:
    """Turn toward the strongest target sector, drive forward, escape on contact.

    Escape escalates: every bump backs off for a few ticks, then turns in place
    for a growing number of ticks, alternating direction, so corners do not
    trap it. Used to exercise the planner layer independently of any learned brain.
    """

    def __init__(self, speed: float = 1.0, turn: float = 0.6, name: str = "follower") -> None:
        self.speed = speed
        self.turn = turn
        self.name = name
        self._backing = 0
        self._turning = 0
        self._bumps = 0

    def reset(self) -> None:
        """Forget the escape counters."""
        self._backing = 0
        self._turning = 0
        self._bumps = 0

    def act(self, channels: Mapping[str, float]) -> Wheels:
        """Simple reactive policy."""
        bumped = channels.get("bumper_left", 0.0) or channels.get("bumper_right", 0.0)
        if bumped and not self._backing and not self._turning:
            self._bumps += 1
            self._backing = 3
            self._turning = min(12, 3 + 2 * self._bumps)
        if self._backing > 0:
            self._backing -= 1
            return -0.6, -0.6
        if self._turning > 0:
            self._turning -= 1
            direction = 1.0 if self._bumps % 2 else -1.0
            return direction * self.speed, -direction * self.speed
        front = channels.get("sensor_front", 0.0)
        if front > 0.7:
            left_closer = channels.get("sensor_left", 0.0) > channels.get("sensor_right", 0.0)
            return (self.speed, -self.speed) if left_closer else (-self.speed, self.speed)
        left, mid, right = (channels.get(f"target_{s}", 0.0) for s in ("left", "front", "right"))
        if mid >= max(left, right) and mid > 0.0:
            self._bumps = max(0, self._bumps - 1)  # progress: relax the escalation
            return self.speed, self.speed
        if left > right:
            return self.speed - self.turn, self.speed
        if right > left:
            return self.speed, self.speed - self.turn
        return self.speed, self.speed
