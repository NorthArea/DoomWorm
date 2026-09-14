"""Neural activity -> motor commands (Plan §2.2, §14)."""

from __future__ import annotations

from collections.abc import Mapping


class MotorAdapter:
    """Read two named neurons as left/right wheel commands in [0, 1]."""

    def __init__(self, left: str, right: str) -> None:
        self.left = left
        self.right = right

    def __call__(self, activity: Mapping[str, float]) -> tuple[float, float]:
        """Return ``(motor_left, motor_right)``."""
        return activity.get(self.left, 0.0), activity.get(self.right, 0.0)
