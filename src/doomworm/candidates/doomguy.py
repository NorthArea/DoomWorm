"""A hand-written Doom player: the zero-learning floor of the Doom benchmark (Plan §20.4, track B).

Reflexes on the structured channels only (Plan §25): exit gradient, enemy
sectors (line of sight), wall rays, bumper, health, ammo. No map, no learning.

    escape    contact (bumper, or a ray at touching range): back off, spin away,
              longer with every repeated bump
    hurt      taking damage without a gun: turn away from the enemy and run
    shoot     a gun and an enemy on the gun line (aim channel) within range: stand and fire
    face      a gun and an enemy on a side: turn toward it; in the front sector but off the
              line: nudge the heading until the aim centres; centred but far: close in
              (but not after two bumps: no chasing through walls, only shots in range)
    avoid     no gun and an enemy ahead: bend away from it
    exit      otherwise follow the exit gradient, veer off walls

Every trained brain has to beat this row on the Doom levels.
"""

from __future__ import annotations

from collections.abc import Mapping

from doomworm.candidates.base import Wheels

FIRE_REACH = 1.0 / 6.0  # 1/distance signal at the gun's range (fire_range 6)
AIM_LOCK = 0.8  # aim channel value (within 6 degrees of the gun line) before firing


class DoomguyBrain:
    """Reflex player with an escalating escape; exposes ``fire`` after every ``act``."""

    def __init__(self, speed: float = 1.0, turn: float = 0.6, name: str = "doomguy") -> None:
        self.speed = speed
        self.turn = turn
        self.name = name
        self.fire = 0.0
        self._last_health: float | None = None
        self._retreat = 0
        self._away: Wheels = (0.0, 0.0)
        self._escape: list[Wheels] = []
        self._bumps = 0
        self._turn_dir = 1.0
        self._last_aim = 0.0

    def reset(self) -> None:
        """Forget every counter."""
        self.fire = 0.0
        self._last_health = None
        self._retreat = 0
        self._escape = []
        self._bumps = 0
        self._turn_dir = 1.0
        self._last_aim = 0.0

    def act(self, channels: Mapping[str, float]) -> Wheels:
        """Reflex priority: escape > retreat when hurt > shoot > face > avoid > exit."""
        self.fire = 0.0
        get = channels.get
        health = get("health", 1.0)
        hurt = self._last_health is not None and health < self._last_health
        self._last_health = health
        e_left, e_front, e_right = (get(f"danger_{s}", 0.0) for s in ("left", "front", "right"))
        s_left, s_front, s_right = (get(f"sensor_{s}", 0.0) for s in ("left", "front", "right"))
        ammo = get("ammo", 0.0) > 0.0  # rounds left, as a fraction of a full magazine
        enemy = max(e_left, e_front, e_right)

        touching = get("bumper_left", 0.0) or get("bumper_right", 0.0) or s_front >= 0.95
        if touching and not self._escape:
            self._bumps += 1
            spin: Wheels = (
                (self.speed, -self.speed) if s_left > s_right else (-self.speed, self.speed)
            )
            self._escape = [(-0.6, -0.6)] * 3 + [spin] * min(12, 3 + 2 * self._bumps)
        if self._escape:
            return self._escape.pop(0)

        if not ammo and hurt and enemy > 0.0 and self._retreat == 0:
            self._retreat = 20  # being hit without a gun: about-face (8 ticks), then run
            self._away = (
                (self.speed, -self.speed) if e_left >= e_right else (-self.speed, self.speed)
            )
        if self._retreat > 0:
            self._retreat -= 1
            return self._away if self._retreat > 12 else (self.speed, self.speed)

        # after two bumps the enemy is not worth chasing through walls: only shoot what is in range
        chase = self._bumps < 2
        if ammo and enemy > 0.0 and (chase or e_front >= FIRE_REACH):
            aim = get("aim", 0.0)
            if e_front >= max(e_left, e_right):
                if aim >= AIM_LOCK and e_front >= FIRE_REACH:
                    self.fire = 1.0
                    self._last_aim = aim
                    return 0.0, 0.0
                if aim < AIM_LOCK:  # in the sector, off the gun line: nudge until it centres
                    if aim < self._last_aim:
                        self._turn_dir = -self._turn_dir
                    self._last_aim = aim
                    return -0.25 * self._turn_dir, 0.25 * self._turn_dir
                if s_front > 0.7:
                    return self._veer(s_left, s_right)
                return self.speed, self.speed  # centred but too far: close in
            self._turn_dir = 1.0 if e_left > e_right else -1.0  # +1 = turn left
            self._last_aim = 0.0
            return -self.turn * self._turn_dir, self.turn * self._turn_dir

        if not ammo and e_front > 0.25:  # within 4 units, dead ahead: sidestep
            return (self.speed, 0.2) if e_left >= e_right else (0.2, self.speed)

        if s_front > 0.7:
            return self._veer(s_left, s_right)
        t_left, t_front, t_right = (get(f"target_{s}", 0.0) for s in ("left", "front", "right"))
        if t_front >= max(t_left, t_right):
            self._bumps = max(0, self._bumps - 1)  # progress: relax the escalation
            return self.speed, self.speed
        if t_left > t_right:
            return self.speed - self.turn, self.speed
        return self.speed, self.speed - self.turn

    def _veer(self, s_left: float, s_right: float) -> Wheels:
        return (self.speed, -self.speed) if s_left > s_right else (-self.speed, self.speed)
