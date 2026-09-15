"""Randomly generated maps (stage 12+) and apartment-like maps (stage 15+)."""

from doomworm.environments.maze.apartment import ApartmentConfig, apartment_world, rooms_visited
from doomworm.environments.maze.generator import MapConfig, random_world

__all__ = ["ApartmentConfig", "MapConfig", "apartment_world", "random_world", "rooms_visited"]
