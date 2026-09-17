"""Neural activity -> motor commands (Plan §2.2, §14)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def _clip(value: float) -> float:
    return max(-1.0, min(1.0, value))


class MotorAdapter:
    """Read two named neurons as left/right wheel commands (stages 1-4)."""

    def __init__(self, left: str, right: str) -> None:
        self.left = left
        self.right = right

    def __call__(self, activity: Mapping[str, float]) -> tuple[float, float]:
        """Return ``(motor_left, motor_right)``."""
        return activity.get(self.left, 0.0), activity.get(self.right, 0.0)


class GroupMotorAdapter:
    """Wheel commands from neuron groups (Plan §14).

    ``drive = gain_drive * (mean(forward) - mean(reversal))``
    ``turn  = gain_turn  * (mean(turn_right) - mean(turn_left))``
    ``left  = clip(drive + turn)``, ``right = clip(drive - turn)`` in [-1, 1].

    A positive ``turn`` speeds up the left wheel, i.e. turns right. Negative
    drive is reversal: both wheels backwards. Activity is expected to be the
    mean over the brain ticks of one environment step (Plan §3.1); the episode
    loop does that averaging.
    """

    def __init__(
        self,
        forward: Sequence[str],
        reversal: Sequence[str],
        turn_left: Sequence[str],
        turn_right: Sequence[str],
        gain_drive: float = 10.0,
        gain_turn: float = 10.0,
    ) -> None:
        for name, group in (("forward", forward), ("reversal", reversal)):
            if not group:
                raise ValueError(f"{name} group must not be empty")
        self.forward = list(forward)
        self.reversal = list(reversal)
        self.turn_left = list(turn_left)
        self.turn_right = list(turn_right)
        self.gain_drive = gain_drive
        self.gain_turn = gain_turn

    @staticmethod
    def _mean(activity: Mapping[str, float], group: Sequence[str]) -> float:
        return sum(activity.get(n, 0.0) for n in group) / len(group) if group else 0.0

    def components(self, activity: Mapping[str, float]) -> dict[str, float]:
        """Group means plus the derived drive and turn, for inspection."""
        fwd = self._mean(activity, self.forward)
        rev = self._mean(activity, self.reversal)
        left = self._mean(activity, self.turn_left)
        right = self._mean(activity, self.turn_right)
        return {
            "forward": fwd,
            "reversal": rev,
            "turn_left": left,
            "turn_right": right,
            "drive": self.gain_drive * (fwd - rev),
            "turn": self.gain_turn * (right - left),
        }

    def __call__(self, activity: Mapping[str, float]) -> tuple[float, float]:
        """Return ``(motor_left, motor_right)`` in [-1, 1]."""
        c = self.components(activity)
        return _clip(c["drive"] + c["turn"]), _clip(c["drive"] - c["turn"])
