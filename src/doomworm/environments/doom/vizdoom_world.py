"""The real Doom behind the same World contract (Plan §24-26, stages B4+).

    ViZDoom game variables + object list  ->  World state (pose, health, enemies)
                                              -> the world's own sector / ray sensing
                                              -> Sensory Adapter -> brain
    wheels (+ trigger)  ->  forward thrust + turn rate (delta buttons) / ATTACK

A :class:`VizdoomWorld` is built from the same seeded layout as the mini-Doom
level (``environments.doom.levels``), written as a PWAD (``wad.py``) and run
by the engine. The engine owns motion, collisions, the monsters, bullets and
damage; the world only mirrors the engine's state and senses it with the same
code as the simulator (structured observations, no framebuffer: Plan §25).

Optional dependency group ``doom`` (``uv sync --group doom``).
"""

from __future__ import annotations

import contextlib
import math
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from doomworm.environments.doom.levels import doom_world
from doomworm.environments.doom.wad import SCALE, write_wad
from doomworm.environments.simple_2d import AgentState, Enemy, Observation, World

VIZDOOM_LEVELS = tuple(f"vizdoom{n}" for n in range(1, 7))
MONSTERS = frozenset({"Zombieman", "ShotgunGuy", "DoomImp", "Demon"})
MOVE, TURN, ATTACK = range(3)  # delta buttons: thrust per tic, degrees per tic; and the trigger
FORWARD_GAIN = 20.0  # thrust at full drive: measured steady speed 6.6 map units = 0.206 u per tic
_VARS = (
    "POSITION_X", "POSITION_Y", "ANGLE", "HEALTH", "HITCOUNT", "KILLCOUNT",
    "DAMAGE_TAKEN", "AMMO2", "DEAD",
)  # fmt: skip

_open_game: Any = None  # one engine per process; a new world closes the previous one
_watch = False  # show the engine window and run at wall-clock speed (doomworm play --watch)


def set_watch(enabled: bool) -> None:
    """Open the engine window for the next world and pace it at 35 tics a second.

    Watching changes nothing the brain sees: the same PLAYER mode, the same
    seeds and the same numbers as the headless benchmark, only drawn and slowed
    down to real time (Plan §24: the engine is the environment, not a viewer).
    """
    global _watch
    _watch = enabled


def watching() -> bool:
    """True when the engine window is on."""
    return _watch


def is_vizdoom_level(maps: str) -> bool:
    """True for ``vizdoom1`` .. ``vizdoom6``."""
    return maps in VIZDOOM_LEVELS


def wad_dir() -> Path:
    """Where generated maps go (regenerated on every build; not project files)."""
    return Path(tempfile.gettempdir()) / "doomworm_vizdoom"


def _new_game(wad: Path, seed: int, timeout: int) -> Any:
    import vizdoom as vzd

    global _open_game
    if _open_game is not None:
        _open_game.close()
        _open_game = None
    game = vzd.DoomGame()
    game.set_doom_scenario_path(str(wad))
    game.set_doom_map("MAP01")
    game.set_window_visible(_watch)
    game.set_sound_enabled(_watch)
    game.set_screen_resolution(
        vzd.ScreenResolution.RES_640X480 if _watch else vzd.ScreenResolution.RES_160X120
    )
    game.set_render_hud(_watch)
    # Delta buttons carry the differential drive: thrust proportional to the mean wheel,
    # heading change exact to the degree per tic (Plan §26's forward / turn set, continuous).
    game.set_available_buttons(
        [
            vzd.Button.MOVE_FORWARD_BACKWARD_DELTA,
            vzd.Button.TURN_LEFT_RIGHT_DELTA,
            vzd.Button.ATTACK,
        ]
    )
    game.set_button_max_value(vzd.Button.MOVE_FORWARD_BACKWARD_DELTA, int(FORWARD_GAIN))
    game.set_button_max_value(vzd.Button.TURN_LEFT_RIGHT_DELTA, 180)
    game.set_available_game_variables([getattr(vzd.GameVariable, v) for v in _VARS])
    game.set_objects_info_enabled(True)
    game.set_episode_start_time(14)  # the gun is holstered for the first 14 tics
    game.set_episode_timeout(timeout)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_doom_skill(3)
    game.set_seed(seed)
    game.init()
    game.new_episode()
    _open_game = game
    return game


class VizdoomWorld(World):
    """A mini-Doom level run by the Doom engine; same sensing, same actions, same reward.

    Args:
        layout: the seeded 2D level the map is written from.
        seed: engine seed (deterministic monsters and bullets).
        tics: engine tics per environment step (one = 1/35 s). Full drive is a
            steady 0.206 world units per tic, the simulator's 0.2; the engine adds
            momentum (about 5 tics to speed, 1.9 units of coasting after release).
        timeout: engine tics before the episode is cut (the loop's own cap is
            the real limit).
        ambush: monsters wait until they see the player (levels with a
            "stationary" enemy); Doom monsters always walk once awake.
        wad: an existing map to run instead of writing one from the layout
            (stage B11 runs a stock Doom scenario, which nobody generated).
        exit_ends: whether reaching the target ends the episode; a stock
            scenario has no exit to reach.
        offset: world units added to every engine position, so a map whose
            coordinates are negative still fits the world's box.
    """

    def __init__(
        self,
        layout: World,
        seed: int,
        level: str = "vizdoom",
        tics: int = 1,
        timeout: int = 20_000,
        ambush: bool = True,
        wad: Path | None = None,
        exit_ends: bool = True,
        offset: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        super().__init__(
            width=layout.width,
            height=layout.height,
            agent=layout.agent,
            obstacles=layout.obstacles,
            walls=layout.walls,
            target=layout.target,
            enemies=[Enemy(e.x, e.y, e.radius, e.health, e.speed) for e in layout.enemies],
            fire_enabled=layout.has_gun,
            ammo=layout.ammo_max,  # the magazine the level starts with (50 on ours, the pistol's)
            exit_ends=exit_ends,
            hunger_rate=0.0,
            seed=seed,
        )
        self.segments = list(layout.segments)  # angled walls of a stock map (B11)
        self.rooms = list(layout.rooms)
        self.gun = layout.has_gun
        self.tics = tics
        self.level = level
        self.offset = offset
        self.wad = (
            wad
            if wad is not None
            else write_wad(layout, wad_dir() / f"{level}_{seed}.wad", ambush=ambush)
        )
        self.game = _new_game(self.wad, seed, timeout)
        self.engine_done = False
        self._dead_ids: set[int] = set()
        self._vars = self._read_vars()
        self._sync()

    # --- engine state -------------------------------------------------------------

    def _read_vars(self) -> dict[str, float]:
        import vizdoom as vzd

        return {v: float(self.game.get_game_variable(getattr(vzd.GameVariable, v))) for v in _VARS}

    def _sync(self) -> None:
        """Mirror pose, health, ammo and the living monsters from the engine."""
        v = self._vars
        ox, oy = self.offset
        self.agent = AgentState(
            x=v["POSITION_X"] / SCALE + ox,
            y=v["POSITION_Y"] / SCALE + oy,
            heading=math.radians(v["ANGLE"]),
        )
        self.health = max(0.0, min(1.0, v["HEALTH"] / 100.0))
        self.ammo = int(v["AMMO2"])
        state = self.game.get_state()
        if state is not None:
            # a dead monster stays in the list as a corpse: drop the ids marked at their kill
            self.enemies = [
                Enemy(o.position_x / SCALE + ox, o.position_y / SCALE + oy, radius=0.5, health=1)
                for o in state.objects
                if o.name in MONSTERS and o.id not in self._dead_ids
            ]

    @property
    def fire_enabled(self) -> bool:
        """The engine owns the ammo count; the gun is loaded while AMMO2 is positive."""
        return self.gun and self.ammo > 0

    @property
    def finished(self) -> bool:
        """Dead, through the exit, or the engine ended the episode (timeout)."""
        return self.dead or self.exited or self.engine_done

    # --- dynamics -------------------------------------------------------------------

    def step(self, motor_left: float, motor_right: float, fire: bool = False) -> Observation:
        """Wheels -> buttons -> one engine step -> events, on the world's own Observation."""
        left = max(-1.0, min(1.0, motor_left))
        right = max(-1.0, min(1.0, motor_right))
        self.last_command = (left, right)
        drive = (left + right) / 2.0
        angular = (right - left) / self.wheel_base * self.speed  # rad per tick, as in the world
        buttons = [0.0] * 3
        buttons[MOVE] = FORWARD_GAIN * drive
        buttons[TURN] = -math.degrees(angular)  # a positive delta turns clockwise
        attack = bool(fire and self.fire_enabled)
        buttons[ATTACK] = float(attack)

        before = self._vars
        x0, y0 = self.agent.x, self.agent.y
        if not self.game.is_episode_finished():
            self.game.make_action(buttons, self.tics)
            if _watch:  # PLAYER mode runs as fast as it can; slow it to Doom's 35 tics a second
                time.sleep(self.tics / 35.0)
        after = self._read_vars()
        self._vars = after
        if self.game.is_episode_finished():
            self.engine_done = True
        self._sync()

        moved = math.dist((x0, y0), (self.agent.x, self.agent.y))
        collided = abs(drive) > 0.2 and moved < 0.001  # pushing a wall: no motion at all
        if collided:
            self.collisions += 1
        # the pistol fires a few tics after the press: count the round when it leaves
        fired = after["AMMO2"] < before["AMMO2"]
        if fired:
            self.shots += 1
        hit = int(after["HITCOUNT"] - before["HITCOUNT"])
        killed = int(after["KILLCOUNT"] - before["KILLCOUNT"])
        self.hits += hit
        self.kills += killed
        if killed:
            self._mark_dead()
        damaged = (
            after["DAMAGE_TAKEN"] > before["DAMAGE_TAKEN"] or after["HEALTH"] < before["HEALTH"]
        )
        if damaged:
            self.damage_taken += 1
        if after["DEAD"] > 0.0:
            self.health = 0.0
        reached = self._reach()
        return replace(
            self.observe(),
            collided=collided,
            reached=reached,
            damaged=damaged,
            fired=fired,
            hit=hit,
            killed=killed,
        )

    def _mark_dead(self) -> None:
        """A kill happened: the monster nearest the aim that is no longer moving is the corpse."""
        state = self.game.get_state()
        if state is None:
            return
        ax, ay = self.agent.x, self.agent.y
        candidates = []
        for o in state.objects:
            if o.name not in MONSTERS or o.id in self._dead_ids:
                continue
            bearing = math.atan2(o.position_y / SCALE - ay, o.position_x / SCALE - ax)
            off = abs(math.remainder(bearing - self.agent.heading, math.tau))
            candidates.append((off, o.id))
        if candidates:
            self._dead_ids.add(min(candidates)[1])

    def observe(self) -> Observation:
        """The world's sensing over the mirrored state; the ammo fraction from the engine."""
        fraction = self.ammo / self.ammo_max if self.gun and self.ammo_max else 0.0
        return replace(super().observe(), ammo=max(0.0, min(1.0, fraction)))

    def close(self) -> None:
        """Stop the engine."""
        global _open_game
        if self.game is _open_game:
            _open_game = None
        self.game.close()

    def __del__(self) -> None:  # pragma: no cover - best effort
        with contextlib.suppress(Exception):  # interpreter shutdown
            self.close()


def vizdoom_world(seed: int, level: str, tics: int = 1) -> VizdoomWorld:
    """``vizdoom<n>``: the mini-Doom level ``doom<n>`` of the same seed, run by the engine."""
    if level not in VIZDOOM_LEVELS:
        raise ValueError(f"level must be one of {VIZDOOM_LEVELS}")
    n = int(level[len("vizdoom") :])
    layout = doom_world(seed, f"doom{n}")
    return VizdoomWorld(layout, seed, level=level, tics=tics, ambush=n < 5)
