"""Needs arbitration (Plan §20.3, stage 20): battery > call > clean.

A plain state machine outside the brain. It decides which goal the planner
layer pursues this tick:

    battery low (<= low + trip)   -> go to the dock and stay until charged (>= full)
                                     (trip = estimated charge the way home costs)
    a call is pending             -> go to the call point, done within ``reach``
    otherwise                     -> coverage (nearest unswept cell)

The dock position is where the episode started, in the odometry frame
(the vacuum starts on its dock).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NeedsArbiter:
    """Priority state machine with hysteresis on the battery."""

    low: float = 0.4
    full: float = 0.95
    reach: float = 1.0
    dock: tuple[float, float] | None = None
    call: tuple[float, float] | None = None
    charging: bool = False
    state: str = "clean"

    def reset(self) -> None:
        """New episode: forget dock and charging state; a pending call (user request) stays."""
        self.dock = None
        self.charging = False
        self.state = "clean"

    def request_call(self, x: float, y: float) -> None:
        """A user calls the robot to a point."""
        self.call = (x, y)

    def decide(
        self, pose: tuple[float, float, float], battery: float, trip: float = 0.0
    ) -> tuple[str, tuple[float, float] | None]:
        """Return ``(state, goal)``; goal None means coverage mode.

        ``trip`` is the charge the way back to the dock is expected to cost; the
        robot leaves for the dock while it still has ``low`` on top of that.
        """
        x, y, _ = pose
        if self.dock is None:
            self.dock = (x, y)
        if self.charging:
            if battery >= self.full:
                self.charging = False
            else:
                self.state = "charge"
                return self.state, self.dock
        if battery <= self.low + trip:
            self.charging = True
            self.state = "charge"
            return self.state, self.dock
        if self.call is not None:
            if abs(x - self.call[0]) <= self.reach and abs(y - self.call[1]) <= self.reach:
                self.call = None
            else:
                self.state = "call"
                return self.state, self.call
        self.state = "clean"
        return self.state, None
