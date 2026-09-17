"""Bounded 2D world with circular obstacles and a differential-drive agent.

Coordinates: x to the right, y up, heading in radians counter-clockwise
from the +x axis.

Obstacle sensors are rays from the agent centre; a reading is
``1 - distance / sensor_range`` clamped to [0, 1], so 1.0 = touching.

Food sensors are a gradient sense, not line-of-sight: the *nearest* food
item contributes ``1 / distance`` (clamped to [0, 1]) to the sector its
bearing falls in - front ``[-30°, 30°]``, left ``(30°, 180°]``, right
``[-180°, -30°)`` - and the other sectors read 0. Tracking a single source
keeps the three channels mutually exclusive.

Hunger grows by ``hunger_rate`` per tick, saturates at 1.0 (starved) and
resets to 0.0 when the agent touches food, which is then consumed. With
``respawn_food`` every eaten item is replaced at a random free spot drawn
from the world's seeded RNG, so the food count stays constant.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, replace

from doomworm.body import Body, DifferentialDrive, Drive


@dataclass(frozen=True)
class Obstacle:
    """Static circular obstacle."""

    x: float
    y: float
    radius: float


@dataclass(frozen=True)
class Food:
    """Food item; consumed on contact."""

    x: float
    y: float
    radius: float = 0.3


@dataclass(frozen=True)
class Target:
    """Goal position ("come to X", Plan §19); sensed like food, reached on contact."""

    x: float
    y: float
    radius: float = 0.3


@dataclass(frozen=True)
class Wall:
    """Axis-aligned solid rectangle (room walls, furniture)."""

    x: float
    y: float
    w: float
    h: float

    @property
    def x1(self) -> float:
        """Right edge."""
        return self.x + self.w

    @property
    def y1(self) -> float:
        """Top edge."""
        return self.y + self.h

    def distance(self, px: float, py: float) -> float:
        """Distance from a point to the rectangle (0 inside)."""
        dx = max(self.x - px, 0.0, px - self.x1)
        dy = max(self.y - py, 0.0, py - self.y1)
        return math.hypot(dx, dy)


@dataclass(frozen=True)
class Danger:
    """Hazard zone (Plan §20): sensed like food, not solid; every tick inside costs health."""

    x: float
    y: float
    radius: float = 1.0


@dataclass
class Enemy:
    """Mini-Doom monster (Plan §21-23, §29-31): sensed like a danger, hurts in range, can be shot.

    Not solid. Every tick the agent is within ``attack_range`` with line of sight it
    loses ``damage`` health. With ``speed`` > 0 it walks toward the agent while it sees
    it (stage B9). ``health`` hits kill it.
    """

    x: float
    y: float
    radius: float = 0.6  # a Doom zombie is 20 map units wide (0.625 u); the body the shot must hit
    health: int = 3
    speed: float = 0.0
    attack_range: float = 3.0
    damage: float = 0.05


@dataclass(frozen=True)
class Dock:
    """Charging dock (Plan §20.3, stage 16): persistent, beacon of limited range."""

    x: float
    y: float
    radius: float = 0.6
    beacon_range: float = 8.0
    charge_rate: float = 0.02  # battery gained per tick while docked (50 ticks to full)


class DirtMap:
    """Grid of cells to clean; a cell is floor if its centre is not inside a solid."""

    def __init__(self, world: World, cell: float = 1.0) -> None:
        self.cell = cell
        self.cols = round(world.width / cell)
        self.rows = round(world.height / cell)
        self.floor: set[tuple[int, int]] = set()
        for i in range(self.cols):
            for j in range(self.rows):
                cx, cy = (i + 0.5) * cell, (j + 0.5) * cell
                if world.clearance(cx, cy) > 0.0:
                    self.floor.add((i, j))
        self.dirty: set[tuple[int, int]] = set(self.floor)

    @property
    def coverage(self) -> float:
        """Share of floor cells cleaned so far."""
        if not self.floor:
            return 0.0
        return 1.0 - len(self.dirty) / len(self.floor)

    def clean_around(self, x: float, y: float, radius: float) -> int:
        """Clean every dirty cell whose centre lies within ``radius``; return the count."""
        cleaned = 0
        i0, i1 = int((x - radius) / self.cell), int((x + radius) / self.cell)
        j0, j1 = int((y - radius) / self.cell), int((y + radius) / self.cell)
        for i in range(max(0, i0), min(self.cols, i1 + 1)):
            for j in range(max(0, j0), min(self.rows, j1 + 1)):
                if (i, j) in self.dirty:
                    cx, cy = (i + 0.5) * self.cell, (j + 0.5) * self.cell
                    if math.dist((x, y), (cx, cy)) <= radius:
                        self.dirty.discard((i, j))
                        cleaned += 1
        return cleaned

    def dirty_centres(self) -> list[tuple[float, float]]:
        """World coordinates of the cells still dirty."""
        return [((i + 0.5) * self.cell, (j + 0.5) * self.cell) for i, j in sorted(self.dirty)]


@dataclass(frozen=True)
class AgentState:
    """Pose of the agent."""

    x: float
    y: float
    heading: float = 0.0


@dataclass(frozen=True)
class Observation:
    """What the agent perceives after a step."""

    sensor_left: float
    sensor_front: float
    sensor_right: float
    food_left: float = 0.0
    food_front: float = 0.0
    food_right: float = 0.0
    hunger: float = 0.0
    target_left: float = 0.0
    target_front: float = 0.0
    target_right: float = 0.0
    danger_left: float = 0.0
    danger_front: float = 0.0
    danger_right: float = 0.0
    health: float = 1.0
    battery: float = 1.0
    dock_left: float = 0.0
    dock_front: float = 0.0
    dock_right: float = 0.0
    docked: bool = False
    cleaned: int = 0
    ammo: float = 0.0
    aim: float = 0.0
    prey_left: float = 0.0  # the same enemy read as prey, not as a hazard (track B)
    prey_front: float = 0.0
    prey_right: float = 0.0
    collided: bool = False
    ate: bool = False
    reached: bool = False
    damaged: bool = False
    fired: bool = False
    hit: int = 0
    killed: int = 0

    def as_channels(self) -> dict[str, float]:
        """Numeric channels for the sensory adapter."""
        return {
            "sensor_left": self.sensor_left,
            "sensor_front": self.sensor_front,
            "sensor_right": self.sensor_right,
            "food_left": self.food_left,
            "food_front": self.food_front,
            "food_right": self.food_right,
            "hunger": self.hunger,
            "target_left": self.target_left,
            "target_front": self.target_front,
            "target_right": self.target_right,
            "danger_left": self.danger_left,
            "danger_front": self.danger_front,
            "danger_right": self.danger_right,
            "health": self.health,
            "battery": self.battery,
            "dock_left": self.dock_left,
            "dock_front": self.dock_front,
            "dock_right": self.dock_right,
            "ammo": self.ammo,
            "aim": self.aim,
            "prey_left": self.prey_left,
            "prey_front": self.prey_front,
            "prey_right": self.prey_right,
        }


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _wrap_angle(angle: float) -> float:
    """Wrap to (-pi, pi]."""
    return math.pi - (math.pi - angle) % (2.0 * math.pi)


FOOD_FRONT_HALF_ANGLE = math.pi / 6  # 30 degrees


class World:
    """Stage-1 environment: one agent, static obstacles, box walls."""

    def __init__(
        self,
        *,
        width: float = 20.0,
        height: float = 20.0,
        agent: AgentState | None = None,
        obstacles: Sequence[Obstacle] = (),
        walls: Sequence[Wall] = (),
        foods: Sequence[Food] = (),
        hunger_rate: float = 0.004,
        respawn_food: bool = False,
        target: Target | None = None,
        respawn_target: bool = False,
        dangers: Sequence[Danger] = (),
        damage_rate: float = 0.25,
        dock: Dock | None = None,
        enemies: Sequence[Enemy] = (),
        fire_enabled: bool = False,
        fire_range: float = 6.0,
        fire_cooldown: int = 5,
        ammo: int = 50,
        exit_ends: bool = False,
        seed: int | None = None,
        agent_radius: float = 0.5,
        sensor_range: float = 4.0,
        sensor_angles: tuple[float, float, float] = (math.pi / 4, 0.0, -math.pi / 4),
        speed: float = 0.2,
        wheel_base: float = 1.0,
    ) -> None:
        self.width = width
        self.height = height
        self.agent = agent if agent is not None else AgentState(x=3.0, y=10.0)
        self.obstacles = list(obstacles)
        self.walls = list(walls)
        # The vehicle this world runs: two wheels unless a preset says otherwise
        # (the car of Plan §20.5.1 is mecanum and can move sideways).
        self.body: Body = DifferentialDrive()
        self.last_drive = Drive()
        # Arbitrary-angle solid lines. Generated levels use axis-aligned walls; a map
        # that comes from the Doom engine (stage B11) has angled ones (Plan §33).
        self.segments: list[tuple[float, float, float, float]] = []
        self.rooms: list[tuple[float, float, float, float]] = []  # optional (x0, y0, x1, y1)
        self.foods = list(foods)
        self.respawn_food = respawn_food
        self.target = target
        self.respawn_target = respawn_target
        self.targets_reached = 0
        self.dangers = list(dangers)
        self.damage_rate = damage_rate
        self.health = 1.0
        self.damage_taken = 0
        self.dock = dock
        # mini-Doom (Plan §21-23): monsters, the gun and the exit
        self.enemies = list(enemies)
        self.has_gun = fire_enabled
        self.ammo_max = ammo
        self.ammo = ammo if fire_enabled else 0
        self.fire_range = fire_range
        self.fire_cooldown = fire_cooldown
        self.exit_ends = exit_ends
        self.shots = 0
        self.hits = 0
        self.kills = 0
        self.exited = False
        self._cooldown = 0
        self.dirt: DirtMap | None = None
        self.charging_ticks = 0
        self.dockings = 0
        self._docked = False
        self.rng = random.Random(seed)
        self.hunger_rate = hunger_rate
        self.hunger = 0.0
        self.food_eaten = 0
        self.agent_radius = agent_radius
        self.sensor_range = sensor_range
        self.sensor_angles = sensor_angles  # (left, front, right) offsets from heading
        self.speed = speed
        self.wheel_base = wheel_base
        self.collisions = 0
        self.last_command: tuple[float, float] = (0.0, 0.0)  # what the motors were told

    @property
    def starved(self) -> bool:
        """True once hunger has saturated."""
        return self.hunger >= 1.0

    @property
    def battery(self) -> float:
        """Charge level: the complement of hunger (Plan §20.3, hunger -> battery)."""
        return 1.0 - self.hunger

    @property
    def coverage(self) -> float:
        """Share of the floor cleaned (0 without a dirt map)."""
        return self.dirt.coverage if self.dirt is not None else 0.0

    @property
    def docked(self) -> bool:
        """True while the agent sits on the dock."""
        return self._docked

    def init_dirt(self, cell: float = 1.0) -> DirtMap:
        """Create the dirt map from the current geometry; call after walls are final."""
        self.dirt = DirtMap(self, cell)
        return self.dirt

    @property
    def dead(self) -> bool:
        """True once health is gone (danger) or hunger has saturated."""
        return self.health <= 0.0 or self.starved

    @property
    def fire_enabled(self) -> bool:
        """True while the agent carries a loaded gun (a Doom pistol starts with 50 rounds)."""
        return self.has_gun and self.ammo > 0

    @property
    def finished(self) -> bool:
        """True when the episode is over: dead, or through the exit (mini-Doom)."""
        return self.dead or self.exited

    # --- dynamics -----------------------------------------------------------

    def step(self, motor_left: float, motor_right: float, fire: bool = False) -> Observation:
        """Apply a differential-drive wheel pair for one tick (the pre-stage-24 call).

        ``fire`` pulls the trigger (Plan §22); it does nothing unless the world has
        ``fire_enabled`` and the cooldown has elapsed.
        """
        left = _clamp(motor_left, -1.0, 1.0)
        right = _clamp(motor_right, -1.0, 1.0)
        return self.drive(Drive.from_wheels(left, right, 1.0 if fire else 0.0))

    def drive(self, intent: Drive) -> Observation:
        """Apply a brain's intent for one tick, through whatever body this world has.

        ``turn`` is half the wheel difference, which is what the motor adapter has
        always produced, so ``angular`` keeps its old value: a wheel pair routed
        through :meth:`Drive.from_wheels` moves the agent exactly as before.
        Sideways motion only happens on a body that has it (Plan §20.5.1).
        """
        intent = intent.clipped()
        left, right = DifferentialDrive().wheels(intent)
        self.last_command = (left, right)
        self.last_drive = intent
        linear = intent.forward * self.speed
        angular = 2.0 * intent.turn / self.wheel_base * self.speed
        lateral = intent.strafe * self.speed if self.body.strafes else 0.0

        heading = self.agent.heading + angular
        x = self.agent.x + math.cos(heading) * linear - math.sin(heading) * lateral
        y = self.agent.y + math.sin(heading) * linear + math.cos(heading) * lateral
        candidate = AgentState(x=x, y=y, heading=heading)
        fire = intent.pulls_trigger

        collided = self._collides(candidate)
        if collided:
            self.collisions += 1
            self.agent = replace(self.agent, heading=heading)  # rotate in place
        else:
            self.agent = candidate

        self.hunger = _clamp(self.hunger + self.hunger_rate, 0.0, 1.0)
        ate = self._eat()
        reached = self._reach()
        fired, hit, killed = self._fire(fire)
        self._move_enemies()
        damaged = self._take_damage()
        cleaned = self._clean()
        docked = self._charge()
        return replace(
            self.observe(),
            collided=collided,
            ate=ate,
            reached=reached,
            damaged=damaged,
            cleaned=cleaned,
            docked=docked,
            fired=fired,
            hit=hit,
            killed=killed,
        )

    # --- mini-Doom (Plan §21-23) ------------------------------------------------

    def line_of_sight(self, x: float, y: float) -> bool:
        """True when no wall or obstacle lies between the agent and the point."""
        ax, ay = self.agent.x, self.agent.y
        distance = math.dist((ax, ay), (x, y))
        if distance <= 1e-9:
            return True
        return self.ray_distance(math.atan2(y - ay, x - ax), limit=distance) >= distance

    def _fire(self, fire: bool) -> tuple[bool, int, int]:
        """Hitscan along the heading (like Doom's pistol): the nearest visible body on the ray."""
        if self._cooldown > 0:
            self._cooldown -= 1
        if not (fire and self.fire_enabled) or self._cooldown > 0:
            return False, 0, 0
        self._cooldown = self.fire_cooldown
        self.ammo -= 1
        self.shots += 1
        ax, ay = self.agent.x, self.agent.y
        on_ray = []
        for e in self.enemies:
            distance = math.dist((ax, ay), (e.x, e.y))
            if distance > self.fire_range:
                continue
            bearing = _wrap_angle(math.atan2(e.y - ay, e.x - ax) - self.agent.heading)
            if abs(bearing) <= math.atan2(e.radius, distance) and self.line_of_sight(e.x, e.y):
                on_ray.append((distance, e))
        if not on_ray:
            return True, 0, 0
        _, target = min(on_ray, key=lambda pair: pair[0])
        target.health -= 1
        self.hits += 1
        if target.health > 0:
            return True, 1, 0
        self.enemies.remove(target)
        self.kills += 1
        return True, 1, 1

    def _move_enemies(self) -> None:
        """A walking enemy closes in on the agent while it sees it; walls stop it."""
        ax, ay = self.agent.x, self.agent.y
        for e in self.enemies:
            if e.speed <= 0.0:
                continue
            distance = math.dist((ax, ay), (e.x, e.y))
            if distance <= e.attack_range / 2.0 or not self.line_of_sight(e.x, e.y):
                continue
            step = min(e.speed, distance)
            nx = e.x + (ax - e.x) / distance * step
            ny = e.y + (ay - e.y) / distance * step
            if self.clearance(nx, ny) > e.radius:
                e.x, e.y = nx, ny

    def spawn_enemy(
        self,
        *,
        radius: float = 0.5,
        health: int = 3,
        speed: float = 0.0,
        margin: float = 1.0,
        min_distance: float = 5.0,
        tries: int = 200,
    ) -> Enemy:
        """Draw an enemy position clear of solids, away from the agent and the exit."""
        for _ in range(tries):
            x = self.rng.uniform(radius + margin, self.width - radius - margin)
            y = self.rng.uniform(radius + margin, self.height - radius - margin)
            if self.clearance(x, y) <= radius + margin:
                continue
            if math.dist((x, y), (self.agent.x, self.agent.y)) <= min_distance:
                continue
            t = self.target
            if t is not None and math.dist((x, y), (t.x, t.y)) <= radius + margin + 1.0:
                continue
            return Enemy(x=x, y=y, radius=radius, health=health, speed=speed)
        raise RuntimeError("could not place enemy after many tries")

    def _clean(self) -> int:
        if self.dirt is None:
            return 0
        brush = self.agent_radius + self.dirt.cell / 2  # the brush is wider than the body
        return self.dirt.clean_around(self.agent.x, self.agent.y, brush)

    def _charge(self) -> bool:
        d = self.dock
        if d is None:
            return False
        docked = math.dist((self.agent.x, self.agent.y), (d.x, d.y)) < d.radius
        if docked:
            if not self._docked:
                self.dockings += 1
            self.charging_ticks += 1
            self.hunger = _clamp(self.hunger - d.charge_rate, 0.0, 1.0)
        self._docked = docked
        return docked

    def _take_damage(self) -> bool:
        ax, ay = self.agent.x, self.agent.y
        inside = any(
            math.dist((ax, ay), (d.x, d.y)) < d.radius + self.agent_radius for d in self.dangers
        )
        loss = self.damage_rate if inside else 0.0
        for e in self.enemies:
            if math.dist((ax, ay), (e.x, e.y)) <= e.attack_range and self.line_of_sight(e.x, e.y):
                loss += e.damage
        if loss <= 0.0:
            return False
        self.health = _clamp(self.health - loss, 0.0, 1.0)
        self.damage_taken += 1
        return True

    def spawn_danger(self, radius: float = 1.0, margin: float = 1.0, tries: int = 100) -> Danger:
        """Draw a danger zone clear of walls, obstacles, the agent and the target."""
        for _ in range(tries):
            x = self.rng.uniform(radius + margin, self.width - radius - margin)
            y = self.rng.uniform(radius + margin, self.height - radius - margin)
            near_obstacle = self.clearance(x, y) <= radius + margin
            near_agent = math.dist((x, y), (self.agent.x, self.agent.y)) <= radius + 2.0
            t = self.target
            near_target = t is not None and math.dist((x, y), (t.x, t.y)) <= radius + margin
            if not (near_obstacle or near_agent or near_target):
                return Danger(x=x, y=y, radius=radius)
        raise RuntimeError("could not place danger after many tries")

    def _reach(self) -> bool:
        t = self.target
        if t is None:
            return False
        if math.dist((self.agent.x, self.agent.y), (t.x, t.y)) >= self.agent_radius + t.radius:
            return False
        self.targets_reached += 1
        if self.exit_ends:
            self.exited = True
        self.target = self.spawn_target(t.radius) if self.respawn_target else None
        return True

    def spawn_target(self, radius: float = 0.3, margin: float = 1.0, tries: int = 100) -> Target:
        """Draw a target position clear of walls, obstacles and the agent."""
        f = self.spawn_food(radius=radius, margin=margin, tries=tries)
        return Target(x=f.x, y=f.y, radius=radius)

    def _eat(self) -> bool:
        reach = self.agent_radius
        remaining = [
            f
            for f in self.foods
            if math.dist((self.agent.x, self.agent.y), (f.x, f.y)) >= reach + f.radius
        ]
        eaten = len(self.foods) - len(remaining)
        if eaten == 0:
            return False
        self.foods = remaining
        self.food_eaten += eaten
        self.hunger = 0.0
        if self.respawn_food:
            for _ in range(eaten):
                self.foods.append(self.spawn_food())
        return True

    def spawn_food(self, radius: float = 0.3, margin: float = 1.0, tries: int = 100) -> Food:
        """Draw a food position clear of walls, obstacles and the agent."""
        for _ in range(tries):
            x = self.rng.uniform(margin, self.width - margin)
            y = self.rng.uniform(margin, self.height - margin)
            near_agent = math.dist((x, y), (self.agent.x, self.agent.y)) <= 2.0
            if self.clearance(x, y) > margin and not near_agent:
                return Food(x=x, y=y, radius=radius)
        raise RuntimeError("could not place food after many tries")

    def _collides(self, state: AgentState) -> bool:
        r = self.agent_radius
        if not (r <= state.x <= self.width - r and r <= state.y <= self.height - r):
            return True
        if any(math.dist((state.x, state.y), (o.x, o.y)) < r + o.radius for o in self.obstacles):
            return True
        return any(w.distance(state.x, state.y) < r for w in self.walls)

    def clearance(self, x: float, y: float) -> float:
        """Distance from a point to the nearest obstacle or wall surface."""
        best = min(x, y, self.width - x, self.height - y)
        for o in self.obstacles:
            best = min(best, math.dist((x, y), (o.x, o.y)) - o.radius)
        for w in self.walls:
            best = min(best, w.distance(x, y))
        return best

    def room_index(self, x: float, y: float) -> int | None:
        """Index of the room containing the point, if rooms are defined."""
        for i, (x0, y0, x1, y1) in enumerate(self.rooms):
            if x0 <= x <= x1 and y0 <= y <= y1:
                return i
        return None

    # --- sensing ------------------------------------------------------------

    def observe(self) -> Observation:
        """Read obstacle sensors, food sensors and hunger."""
        left, front, right = (self._sense(offset) for offset in self.sensor_angles)
        f_left, f_front, f_right = self._sense_food()
        t_left, t_front, t_right = self._sense_target()
        d_left, d_front, d_right = self._sense_danger()
        p_left, p_front, p_right = self._sense_prey()
        k_left, k_front, k_right = self._sense_dock()
        return Observation(
            sensor_left=left,
            sensor_front=front,
            sensor_right=right,
            aim=self._aim(),
            food_left=f_left,
            food_front=f_front,
            food_right=f_right,
            hunger=self.hunger,
            target_left=t_left,
            target_front=t_front,
            target_right=t_right,
            danger_left=d_left,
            danger_front=d_front,
            danger_right=d_right,
            prey_left=p_left,
            prey_front=p_front,
            prey_right=p_right,
            health=self.health,
            battery=self.battery,
            dock_left=k_left,
            dock_front=k_front,
            dock_right=k_right,
            docked=self._docked,
            ammo=self.ammo / self.ammo_max if self.has_gun and self.ammo_max else 0.0,
        )

    def _sense_dock(self) -> tuple[float, float, float]:
        d = self.dock
        if d is None:
            return 0.0, 0.0, 0.0
        if math.dist((self.agent.x, self.agent.y), (d.x, d.y)) > d.beacon_range:
            return 0.0, 0.0, 0.0
        return self._sector_signal(d.x, d.y)

    def _aim(self) -> float:
        """How centred the nearest visible enemy is: 1 dead ahead, 0 at the sector edge or none.

        The one extra structured number a shooter needs (Plan §25, §30): the sectors
        say "ahead", this says how far off the gun line, without a framebuffer.
        """
        ax, ay = self.agent.x, self.agent.y
        best = 0.0
        best_distance = math.inf
        for e in self.enemies:
            distance = math.dist((ax, ay), (e.x, e.y))
            if distance >= best_distance or not self.line_of_sight(e.x, e.y):
                continue
            bearing = _wrap_angle(math.atan2(e.y - ay, e.x - ax) - self.agent.heading)
            if abs(bearing) <= FOOD_FRONT_HALF_ANGLE:
                best_distance = distance
                best = 1.0 - abs(bearing) / FOOD_FRONT_HALF_ANGLE
        return best

    def _sense_danger(self) -> tuple[float, float, float]:
        """Nearest hazard: a danger zone (a smell, through walls) or a visible monster.

        The enemy is a danger under another name, but it is *seen*: a monster
        behind a wall is not on the channels (Plan §2.4, §25).
        """
        points = [(d.x, d.y) for d in self.dangers]
        points += [(e.x, e.y) for e in self.enemies if self.line_of_sight(e.x, e.y)]
        if not points:
            return 0.0, 0.0, 0.0
        ax, ay = self.agent.x, self.agent.y
        nx, ny = min(points, key=lambda p: math.dist((ax, ay), p))
        return self._sector_signal(nx, ny)

    def _sense_prey(self) -> tuple[float, float, float]:
        """The nearest visible enemy read as an attractant, by side (track B, stage B2d).

        The same monster already arrives on ``danger_*``, which the default mapping
        routes to the nociceptive pair ASH -- the animal's escape pathway. A brain
        that must line its body up with a target needs the opposite reading as well:
        prey on the left, prey ahead, prey on the right, on the gradient the worm's
        own taxis machinery knows how to climb. Without a loaded gun there is no
        prey, only a hazard.
        """
        if not self.has_gun:
            return 0.0, 0.0, 0.0
        ax, ay = self.agent.x, self.agent.y
        seen = [e for e in self.enemies if self.line_of_sight(e.x, e.y)]
        if not seen:
            return 0.0, 0.0, 0.0
        nearest = min(seen, key=lambda e: math.dist((ax, ay), (e.x, e.y)))
        return self._sector_signal(nearest.x, nearest.y)

    def _sense_food(self) -> tuple[float, float, float]:
        if not self.foods:
            return 0.0, 0.0, 0.0
        ax, ay = self.agent.x, self.agent.y
        nearest = min(self.foods, key=lambda f: math.dist((ax, ay), (f.x, f.y)))
        return self._sector_signal(nearest.x, nearest.y)

    def _sense_target(self) -> tuple[float, float, float]:
        if self.target is None:
            return 0.0, 0.0, 0.0
        return self._sector_signal(self.target.x, self.target.y)

    def _sector_signal(self, x: float, y: float) -> tuple[float, float, float]:
        """1/distance gradient of a point source routed to (left, front, right)."""
        ax, ay = self.agent.x, self.agent.y
        dx, dy = x - ax, y - ay
        distance = math.hypot(dx, dy)
        signal = 1.0 if distance <= 1.0 else 1.0 / distance
        bearing = _wrap_angle(math.atan2(dy, dx) - self.agent.heading)
        if abs(bearing) <= FOOD_FRONT_HALF_ANGLE:
            return 0.0, signal, 0.0
        if bearing > 0.0:
            return signal, 0.0, 0.0
        return 0.0, 0.0, signal

    def _sense(self, angle_offset: float) -> float:
        distance = self.ray_distance(self.agent.heading + angle_offset)
        return _clamp(1.0 - distance / self.sensor_range, 0.0, 1.0)

    def ray_distance(self, angle: float, limit: float | None = None) -> float:
        """Distance from the agent centre to the nearest surface along ``angle``.

        Capped at ``limit`` (default: the sensor range); line-of-sight checks pass
        the distance to the point they test.
        """
        ox, oy = self.agent.x, self.agent.y
        dx, dy = math.cos(angle), math.sin(angle)
        best = self.sensor_range if limit is None else limit

        for o in self.obstacles:
            t = _ray_circle(ox, oy, dx, dy, o.x, o.y, o.radius)
            if t is not None:
                best = min(best, t)
        for w in self.walls:
            t = _ray_rect(ox, oy, dx, dy, w)
            if t is not None:
                best = min(best, t)
        for seg in self.segments:
            t = _ray_segment(ox, oy, dx, dy, seg)
            if t is not None:
                best = min(best, t)

        for t in (
            _ray_line(ox, dx, 0.0),
            _ray_line(ox, dx, self.width),
            _ray_line(oy, dy, 0.0),
            _ray_line(oy, dy, self.height),
        ):
            if t is not None:
                best = min(best, t)
        return best


def _ray_segment(
    ox: float,
    oy: float,
    dx: float,
    dy: float,
    segment: tuple[float, float, float, float],
) -> float | None:
    """Distance along the ray to a solid line, or None when it does not cross it."""
    ax, ay, bx, by = segment
    sx, sy = bx - ax, by - ay
    denom = dx * sy - dy * sx
    if abs(denom) < 1e-12:  # parallel
        return None
    wx, wy = ax - ox, ay - oy
    t = (wx * sy - sx * wy) / denom
    u = (wx * dy - dx * wy) / denom
    if t < 0.0 or not 0.0 <= u <= 1.0:
        return None
    return t


def _ray_circle(
    ox: float, oy: float, dx: float, dy: float, cx: float, cy: float, r: float
) -> float | None:
    """Smallest non-negative t with |o + t*d - c| = r, or None."""
    fx, fy = ox - cx, oy - cy
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return None
    root = math.sqrt(disc)
    for t in ((-b - root) / 2.0, (-b + root) / 2.0):
        if t >= 0.0:
            return t
    return None


def _ray_rect(ox: float, oy: float, dx: float, dy: float, rect: Wall) -> float | None:
    """Slab test: smallest non-negative t where the ray enters ``rect``, or None."""
    tmin, tmax = 0.0, math.inf
    for o, d, lo, hi in ((ox, dx, rect.x, rect.x1), (oy, dy, rect.y, rect.y1)):
        if abs(d) < 1e-12:
            if o < lo or o > hi:
                return None
            continue
        t1, t2 = (lo - o) / d, (hi - o) / d
        tmin, tmax = max(tmin, min(t1, t2)), min(tmax, max(t1, t2))
        if tmin > tmax:
            return None
    return tmin


def _ray_line(origin: float, direction: float, line: float) -> float | None:
    """T at which an axis-aligned line ``coord = line`` is crossed, or None."""
    if abs(direction) < 1e-12:
        return None
    t = (line - origin) / direction
    return t if t >= 0.0 else None
