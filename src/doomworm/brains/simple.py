"""Tiny hand-written brains: platform test drivers, not candidates."""

from __future__ import annotations

from collections.abc import Mapping

from doomworm.brains.base import Wheels


class GradientFollower:
    """Turn toward the strongest target sector, drive forward, back off on contact.

    Used to exercise the planner layer independently of any learned brain.
    """

    def __init__(self, speed: float = 1.0, turn: float = 0.6, name: str = "follower") -> None:
        self.speed = speed
        self.turn = turn
        self.name = name
        self._backing = 0

    def reset(self) -> None:
        """Forget the backing-off counter."""
        self._backing = 0

    def act(self, channels: Mapping[str, float]) -> Wheels:
        """Simple reactive policy."""
        if channels.get("bumper_left", 0.0) or channels.get("bumper_right", 0.0):
            self._backing = 4
        if self._backing > 0:
            self._backing -= 1
            return (-0.6, -0.3) if self._backing % 2 else (-0.3, -0.6)
        front = channels.get("sensor_front", 0.0)
        if front > 0.7:
            return (
                (self.speed, -self.speed)
                if channels.get("sensor_left", 0) > channels.get("sensor_right", 0)
                else (-self.speed, self.speed)
            )
        left, mid, right = (channels.get(f"target_{s}", 0.0) for s in ("left", "front", "right"))
        if mid >= max(left, right) and mid > 0.0:
            return self.speed, self.speed
        if left > right:
            return self.speed - self.turn, self.speed
        if right > left:
            return self.speed, self.speed - self.turn
        return self.speed, self.speed
