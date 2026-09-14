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
    collided: bool = False
    ate: bool = False

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
        foods: Sequence[Food] = (),
        hunger_rate: float = 0.004,
        respawn_food: bool = False,
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
        self.foods = list(foods)
        self.respawn_food = respawn_food
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
        return replace(self.observe(), collided=collided, ate=ate)

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
            near_obstacle = any(
                math.dist((x, y), (o.x, o.y)) <= o.radius + margin for o in self.obstacles
            )
            near_agent = math.dist((x, y), (self.agent.x, self.agent.y)) <= 2.0
            if not near_obstacle and not near_agent:
                return Food(x=x, y=y, radius=radius)
        raise RuntimeError("could not place food after many tries")

    def _collides(self, state: AgentState) -> bool:
        r = self.agent_radius
        if not (r <= state.x <= self.width - r and r <= state.y <= self.height - r):
            return True
        return any(math.dist((state.x, state.y), (o.x, o.y)) < r + o.radius for o in self.obstacles)

    # --- sensing ------------------------------------------------------------

    def observe(self) -> Observation:
        """Read obstacle sensors, food sensors and hunger."""
        left, front, right = (self._sense(offset) for offset in self.sensor_angles)
        f_left, f_front, f_right = self._sense_food()
        return Observation(
            sensor_left=left,
            sensor_front=front,
            sensor_right=right,
            food_left=f_left,
            food_front=f_front,
            food_right=f_right,
            hunger=self.hunger,
        )

    def _sense_food(self) -> tuple[float, float, float]:
        if not self.foods:
            return 0.0, 0.0, 0.0
        ax, ay = self.agent.x, self.agent.y
        nearest = min(self.foods, key=lambda f: math.dist((ax, ay), (f.x, f.y)))
        dx, dy = nearest.x - ax, nearest.y - ay
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


def _ray_line(origin: float, direction: float, line: float) -> float | None:
    """T at which an axis-aligned line ``coord = line`` is crossed, or None."""
    if abs(direction) < 1e-12:
        return None
    t = (line - origin) / direction
    return t if t >= 0.0 else None
