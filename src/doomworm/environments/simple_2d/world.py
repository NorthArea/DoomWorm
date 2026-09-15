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
    collided: bool = False
    ate: bool = False
    reached: bool = False
    damaged: bool = False

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

    @property
    def starved(self) -> bool:
        """True once hunger has saturated."""
        return self.hunger >= 1.0

    @property
    def dead(self) -> bool:
        """True once health is gone (danger) or hunger has saturated."""
        return self.health <= 0.0 or self.starved

    # --- dynamics -----------------------------------------------------------

    def step(self, motor_left: float, motor_right: float) -> Observation:
        """Apply motor commands in [-1, 1] (negative = reverse) for one tick."""
        left = _clamp(motor_left, -1.0, 1.0)
        right = _clamp(motor_right, -1.0, 1.0)
        linear = (left + right) / 2.0 * self.speed
        angular = (right - left) / self.wheel_base * self.speed

        heading = self.agent.heading + angular
        x = self.agent.x + math.cos(heading) * linear
        y = self.agent.y + math.sin(heading) * linear
        candidate = AgentState(x=x, y=y, heading=heading)

        collided = self._collides(candidate)
        if collided:
            self.collisions += 1
            self.agent = replace(self.agent, heading=heading)  # rotate in place
        else:
            self.agent = candidate

        self.hunger = _clamp(self.hunger + self.hunger_rate, 0.0, 1.0)
        ate = self._eat()
        reached = self._reach()
        damaged = self._take_damage()
        return replace(self.observe(), collided=collided, ate=ate, reached=reached, damaged=damaged)

    def _take_damage(self) -> bool:
        ax, ay = self.agent.x, self.agent.y
        inside = any(
            math.dist((ax, ay), (d.x, d.y)) < d.radius + self.agent_radius for d in self.dangers
        )
        if not inside:
            return False
        self.health = _clamp(self.health - self.damage_rate, 0.0, 1.0)
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
        return Observation(
            sensor_left=left,
            sensor_front=front,
            sensor_right=right,
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
            health=self.health,
        )

    def _sense_danger(self) -> tuple[float, float, float]:
        if not self.dangers:
            return 0.0, 0.0, 0.0
        ax, ay = self.agent.x, self.agent.y
        nearest = min(self.dangers, key=lambda d: math.dist((ax, ay), (d.x, d.y)))
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

    def ray_distance(self, angle: float) -> float:
        """Distance from the agent centre to the nearest surface along ``angle``."""
        ox, oy = self.agent.x, self.agent.y
        dx, dy = math.cos(angle), math.sin(angle)
        best = self.sensor_range

        for o in self.obstacles:
            t = _ray_circle(ox, oy, dx, dy, o.x, o.y, o.radius)
            if t is not None:
                best = min(best, t)
        for w in self.walls:
            t = _ray_rect(ox, oy, dx, dy, w)
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
