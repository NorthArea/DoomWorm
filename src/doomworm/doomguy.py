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
    veer      a wall close ahead: turn along it
    exit      otherwise follow the exit gradient

Every trained brain has to beat this row on the Doom levels.

Stage 25 found that it is the only thing in the project that ever leaves the
band of flailing at random, which makes *what it knows* the question. So each
reflex is a method returning an intent or `None`, and `ablate` switches any of
them off: the cost of removing one is what that piece of knowledge is worth.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

from wormlab.candidates.base import Wheels

FIRE_REACH = 1.0 / 6.0  # 1/distance signal at the gun's range (fire_range 6)
AIM_LOCK = 0.8  # aim channel value (within 6 degrees of the gun line) before firing
Get = Callable[[str, float], float]
REFLEXES = ("escape", "hurt", "shoot", "face", "avoid", "veer", "exit")


class DoomguyBrain:
    """Reflex player with an escalating escape; exposes ``fire`` after every ``act``."""

    def __init__(
        self,
        speed: float = 1.0,
        turn: float = 0.6,
        name: str = "doomguy",
        ablate: Iterable[str] = (),
    ) -> None:
        self.speed = speed
        self.turn = turn
        self.name = name
        self.ablate = frozenset(ablate)
        unknown = self.ablate - set(REFLEXES)
        if unknown:
            raise ValueError(f"no such reflex: {', '.join(sorted(unknown))}")
        self.fire = 0.0
        self._last_health: float | None = None
        self._retreat = 0
        self._away: Wheels = (0.0, 0.0)
        self._escape_queue: list[Wheels] = []
        self._bumps = 0
        self._turn_dir = 1.0
        self._last_aim = 0.0

    def reset(self) -> None:
        """Forget every counter."""
        self.fire = 0.0
        self._last_health = None
        self._retreat = 0
        self._escape_queue = []
        self._bumps = 0
        self._turn_dir = 1.0
        self._last_aim = 0.0

    def act(self, channels: Mapping[str, float]) -> Wheels:
        """Reflex priority: escape > retreat when hurt > shoot > face > avoid > veer > exit.

        Each reflex either claims the tick or passes. An ablated one always
        passes, and the tick falls through to whatever would have handled it
        next -- which is what makes the cost of removing it readable.
        """
        self.fire = 0.0
        get = channels.get
        health = get("health", 1.0)
        hurt = self._last_health is not None and health < self._last_health
        self._last_health = health

        for name in REFLEXES:
            if name in self.ablate:
                continue
            reflex: Callable[[Get, bool], Wheels | None] = getattr(self, f"_{name}")
            wheels = reflex(get, hurt)
            if wheels is not None:
                return wheels
        return self.speed, self.speed  # every reflex passed: keep going forward

    # --- the reflexes, in priority order ------------------------------------

    def _escape(self, get: Get, hurt: bool) -> Wheels | None:
        """Contact: back off and spin away, longer with every repeated bump."""
        del hurt
        s_left, s_front, s_right = self._rays(get)
        touching = bool(get("bumper_left", 0.0) or get("bumper_right", 0.0) or s_front >= 0.95)
        if touching and not self._escape_queue:
            self._bumps += 1
            spin: Wheels = (
                (self.speed, -self.speed) if s_left > s_right else (-self.speed, self.speed)
            )
            self._escape_queue = [(-0.6, -0.6)] * 3 + [spin] * min(12, 3 + 2 * self._bumps)
        if self._escape_queue:
            return self._escape_queue.pop(0)
        return None

    def _hurt(self, get: Get, hurt: bool) -> Wheels | None:
        """Taking damage with no gun: about-face, then run."""
        e_left, _e_front, e_right = self._sectors(get)
        enemy = max(e_left, _e_front, e_right)
        ammo = get("ammo", 0.0) > 0.0
        if not ammo and hurt and enemy > 0.0 and self._retreat == 0:
            self._retreat = 20  # about-face (8 ticks), then run
            self._away = (
                (self.speed, -self.speed) if e_left >= e_right else (-self.speed, self.speed)
            )
        if self._retreat > 0:
            self._retreat -= 1
            return self._away if self._retreat > 12 else (self.speed, self.speed)
        return None

    def _shoot(self, get: Get, hurt: bool) -> Wheels | None:
        """An enemy on the gun line and in range: stand still and fire."""
        del hurt
        if not self._engaging(get):
            return None
        e_left, e_front, e_right = self._sectors(get)
        if e_front < max(e_left, e_right):
            return None
        aim = get("aim", 0.0)
        if aim >= AIM_LOCK and e_front >= FIRE_REACH:
            self.fire = 1.0
            self._last_aim = aim
            return 0.0, 0.0
        return None

    def _face(self, get: Get, hurt: bool) -> Wheels | None:
        """Come round onto the enemy, centre the gun line, then close the distance."""
        del hurt
        if not self._engaging(get):
            return None
        e_left, e_front, e_right = self._sectors(get)
        s_left, s_front, s_right = self._rays(get)
        if e_front >= max(e_left, e_right):
            aim = get("aim", 0.0)
            if aim < AIM_LOCK:  # in the sector, off the gun line: nudge until it centres
                if aim < self._last_aim:
                    self._turn_dir = -self._turn_dir
                self._last_aim = aim
                return -0.25 * self._turn_dir, 0.25 * self._turn_dir
            if s_front > 0.7:
                return self._veer_from(s_left, s_right)
            return self.speed, self.speed  # centred but too far: close in
        self._turn_dir = 1.0 if e_left > e_right else -1.0  # +1 = turn left
        self._last_aim = 0.0
        return -self.turn * self._turn_dir, self.turn * self._turn_dir

    def _avoid(self, get: Get, hurt: bool) -> Wheels | None:
        """No gun and an enemy close ahead: sidestep."""
        del hurt
        e_left, e_front, e_right = self._sectors(get)
        ammo = get("ammo", 0.0) > 0.0
        if not ammo and e_front > 0.25:  # within 4 units, dead ahead
            return (self.speed, 0.2) if e_left >= e_right else (0.2, self.speed)
        return None

    def _veer(self, get: Get, hurt: bool) -> Wheels | None:
        """A wall close ahead: turn along it."""
        del hurt
        s_left, s_front, s_right = self._rays(get)
        if s_front > 0.7:
            return self._veer_from(s_left, s_right)
        return None

    def _exit(self, get: Get, hurt: bool) -> Wheels | None:
        """Follow the exit gradient."""
        del hurt
        t_left = get("target_left", 0.0)
        t_front = get("target_front", 0.0)
        t_right = get("target_right", 0.0)
        if t_front >= max(t_left, t_right):
            self._bumps = max(0, self._bumps - 1)  # progress: relax the escalation
            return self.speed, self.speed
        if t_left > t_right:
            return self.speed - self.turn, self.speed
        return self.speed, self.speed - self.turn

    # --- shared reads -------------------------------------------------------

    def _engaging(self, get: Get) -> bool:
        """Gun, an enemy, and either room to chase or a shot already in range."""
        e_left, e_front, e_right = self._sectors(get)
        ammo = get("ammo", 0.0) > 0.0
        # after two bumps the enemy is not worth chasing through walls
        chase = self._bumps < 2
        return bool(
            ammo and max(e_left, e_front, e_right) > 0.0 and (chase or e_front >= FIRE_REACH)
        )

    @staticmethod
    def _sectors(get: Get) -> tuple[float, float, float]:
        """Enemy line-of-sight, left / front / right."""
        return get("danger_left", 0.0), get("danger_front", 0.0), get("danger_right", 0.0)

    @staticmethod
    def _rays(get: Get) -> tuple[float, float, float]:
        """Wall rays, left / front / right."""
        return get("sensor_left", 0.0), get("sensor_front", 0.0), get("sensor_right", 0.0)

    def _veer_from(self, s_left: float, s_right: float) -> Wheels:
        return (self.speed, -self.speed) if s_left > s_right else (-self.speed, self.speed)
