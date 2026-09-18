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
    ``turn  = gain_turn  * (mean(turn_right) - mean(turn_left)) * damping``
    ``damping = 1 - aim_damping * mean(aim_group)``, floored at 0
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
        aim_group: Sequence[str] = (),
        aim_damping: float = 0.0,
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
        # Slowing the turn as the target comes onto the line: without it the body
        # sweeps past the few degrees that count (stage 18). The animal does the
        # same thing on food, through the dopaminergic head mechanosensors.
        self.aim_group = list(aim_group)
        self.aim_damping = aim_damping

    @staticmethod
    def _mean(activity: Mapping[str, float], group: Sequence[str]) -> float:
        return sum(activity.get(n, 0.0) for n in group) / len(group) if group else 0.0

    def components(self, activity: Mapping[str, float]) -> dict[str, float]:
        """Group means plus the derived drive and turn, for inspection."""
        fwd = self._mean(activity, self.forward)
        rev = self._mean(activity, self.reversal)
        left = self._mean(activity, self.turn_left)
        right = self._mean(activity, self.turn_right)
        aim = self._mean(activity, self.aim_group) if self.aim_group else 0.0
        damping = max(0.0, 1.0 - self.aim_damping * aim)
        return {
            "forward": fwd,
            "reversal": rev,
            "turn_left": left,
            "turn_right": right,
            "aim": aim,
            "drive": self.gain_drive * (fwd - rev),
            "turn": self.gain_turn * (right - left) * damping,
        }

    def __call__(self, activity: Mapping[str, float]) -> tuple[float, float]:
        """Return ``(motor_left, motor_right)`` in [-1, 1]."""
        c = self.components(activity)
        return _clip(c["drive"] + c["turn"]), _clip(c["drive"] - c["turn"])


class FireAdapter:
    """The trigger from a neuron group (Plan §22): ``fire = gain * mean(group)`` in [0, 1].

    An admitted artifice: the worm has no gun. The group is read like a motor
    group; the episode loop pulls the trigger when the value exceeds 0.5. The
    topology is never changed for it.
    """

    def __init__(self, group: Sequence[str], gain: float = 10.0) -> None:
        if not group:
            raise ValueError("fire group must not be empty")
        self.group = list(group)
        self.gain = gain

    def __call__(self, activity: Mapping[str, float]) -> float:
        """Trigger level in [0, 1]."""
        mean = sum(activity.get(n, 0.0) for n in self.group) / len(self.group)
        return max(0.0, min(1.0, self.gain * mean))
