"""Bounded 2D world with circular obstacles and a differential-drive agent.

Coordinates: x to the right, y up, heading in radians counter-clockwise
from the +x axis. Sensors are rays from the agent centre; a reading is
``1 - distance / sensor_range`` clamped to [0, 1], so 1.0 = touching.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Obstacle:
    """Static circular obstacle."""

    x: float
    y: float
    radius: float


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
    collided: bool = False

    def as_channels(self) -> dict[str, float]:
        """Numeric channels for the sensory adapter."""
        return {
            "sensor_left": self.sensor_left,
            "sensor_front": self.sensor_front,
            "sensor_right": self.sensor_right,
        }


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class World:
    """Stage-1 environment: one agent, static obstacles, box walls."""

    def __init__(
        self,
        *,
        width: float = 20.0,
        height: float = 20.0,
        agent: AgentState | None = None,
        obstacles: Sequence[Obstacle] = (),
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
        self.agent_radius = agent_radius
        self.sensor_range = sensor_range
        self.sensor_angles = sensor_angles  # (left, front, right) offsets from heading
        self.speed = speed
        self.wheel_base = wheel_base
        self.collisions = 0

    # --- dynamics -----------------------------------------------------------

    def step(self, motor_left: float, motor_right: float) -> Observation:
        """Apply motor commands in [0, 1] for one tick and return the observation."""
        left = _clamp(motor_left, 0.0, 1.0)
        right = _clamp(motor_right, 0.0, 1.0)
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
        return replace(self.observe(), collided=collided)

    def _collides(self, state: AgentState) -> bool:
        r = self.agent_radius
        if not (r <= state.x <= self.width - r and r <= state.y <= self.height - r):
            return True
        return any(math.dist((state.x, state.y), (o.x, o.y)) < r + o.radius for o in self.obstacles)

    # --- sensing ------------------------------------------------------------

    def observe(self) -> Observation:
        """Read the three proximity sensors."""
        left, front, right = (self._sense(offset) for offset in self.sensor_angles)
        return Observation(sensor_left=left, sensor_front=front, sensor_right=right)

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
