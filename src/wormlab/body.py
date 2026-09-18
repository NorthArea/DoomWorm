"""What a brain wants, and what the vehicle does about it (Plan §2.2, §3.3).

Until stage 24 every brain returned two wheel commands, because the first
platform was a differential-drive vacuum. That put a vehicle into the brain's
contract: a Doom player has no wheels, and the car of Plan §20.5.1 has four
mecanum ones that can also move sideways.

The contract is now vehicle-neutral. A brain returns a :class:`Drive` -- how
much it wants to go forward, turn, strafe, and whether it pulls the trigger --
and a *body* turns that into whatever actuators exist:

    Drive(forward, turn, strafe, fire)
        -> DifferentialDrive  -> (left, right)          two wheels
        -> MecanumDrive       -> (fl, fr, rl, rr)       four wheels, sideways
        -> DoomBody           -> engine buttons

A body that cannot strafe ignores the component, and a brain with no way to ask
for it leaves it at zero: the worm has no lateral gait, so its strafe is always
zero, and that is a fact about the animal, not a gap in the platform.

The differential body reproduces the old arithmetic exactly -- the wheels are
mixed and *then* clipped, as before -- so every published row still replays.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol, cast


def _clip(value: float) -> float:
    return max(-1.0, min(1.0, value))


@dataclass(frozen=True)
class Drive:
    """A brain's intent, in units no vehicle owns.

    ``forward`` +1 is full ahead, -1 full back. ``turn`` and ``strafe`` are
    positive to the left. ``fire`` is the trigger level; the loop pulls it above
    0.5. Every component is in [-1, 1].
    """

    forward: float = 0.0
    turn: float = 0.0
    strafe: float = 0.0
    fire: float = 0.0

    def clipped(self) -> Drive:
        """The same intent with every component inside [-1, 1]."""
        return Drive(_clip(self.forward), _clip(self.turn), _clip(self.strafe), _clip(self.fire))

    def with_fire(self, fire: float) -> Drive:
        """The same intent with the trigger replaced."""
        return replace(self, fire=fire)

    @property
    def pulls_trigger(self) -> bool:
        """The loop's threshold (Plan §22)."""
        return self.fire > 0.5

    @classmethod
    def from_wheels(cls, left: float, right: float, fire: float = 0.0) -> Drive:
        """Read a differential-drive pair as an intent (the pre-stage-24 brains).

        Exact inverse of :meth:`DifferentialDrive.wheels` while the wheels are
        inside [-1, 1], which is where a brain's output already is.
        """
        return cls(forward=(left + right) / 2.0, turn=(right - left) / 2.0, fire=fire)


class Body(Protocol):
    """A vehicle: it knows which actuators exist and how the intent reaches them."""

    name: str
    strafes: bool

    def wheels(self, drive: Drive) -> tuple[float, ...]:
        """Actuator commands for this intent."""
        ...


class DifferentialDrive:
    """Two wheels, no sideways motion: the vacuum, and the car driven tank-style.

    ``left = clip(forward - turn)``, ``right = clip(forward + turn)``: the mix
    happens first and the clip second, which is what the motor adapter did
    before the body layer existed.
    """

    name = "differential"
    strafes = False

    def wheels(self, drive: Drive) -> tuple[float, float]:
        """``(left, right)`` in [-1, 1]."""
        return _clip(drive.forward - drive.turn), _clip(drive.forward + drive.turn)


class MecanumDrive:
    """Four mecanum wheels (Plan §20.5.1): the same intent plus sideways motion.

    Standard mixing, front-left, front-right, rear-left, rear-right:

        fl = forward - turn + strafe      fr = forward + turn - strafe
        rl = forward - turn - strafe      rr = forward + turn + strafe

    Each wheel is clipped on its own, so a saturated strafe eats into the turn
    exactly as it does on the real drive.
    """

    name = "mecanum"
    strafes = True

    def wheels(self, drive: Drive) -> tuple[float, float, float, float]:
        """``(front_left, front_right, rear_left, rear_right)`` in [-1, 1]."""
        f, t, s = drive.forward, drive.turn, drive.strafe
        return (_clip(f - t + s), _clip(f + t - s), _clip(f - t - s), _clip(f + t + s))

    @staticmethod
    def as_differential(wheels: tuple[float, float, float, float]) -> tuple[float, float]:
        """The pair a two-wheel world sees: each side's mean (the simulator is 2D)."""
        fl, fr, rl, rr = wheels
        return (fl + rl) / 2.0, (fr + rr) / 2.0


BODIES: dict[str, type[DifferentialDrive] | type[MecanumDrive]] = {
    "differential": DifferentialDrive,
    "mecanum": MecanumDrive,
}


def build_body(name: str) -> Body:
    """Body by name."""
    if name not in BODIES:
        raise ValueError(f"unknown body {name!r}, choose from {sorted(BODIES)}")
    return BODIES[name]()


def drive_of(value: object, fire: float = 0.0) -> Drive:
    """Whatever a brain returned, as a :class:`Drive`.

    A brain from before stage 24 returns a wheel pair and carries its trigger on
    the side; a brain written after it returns the intent directly.
    """
    if isinstance(value, Drive):
        return value if value.fire or not fire else value.with_fire(fire)
    pair = cast("Sequence[float]", value)
    return Drive.from_wheels(float(pair[0]), float(pair[1]), fire)
