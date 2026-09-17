"""The engineered layer between the sensors and the brain (Plan §3.2).

Map (occupancy grid), path and coverage planning, needs arbitration, the dock
autopilot / bumper reflex / marker search wrapper (:class:`PlannerLayer`) and
the scripted gradient follower that drives the autopilot. No learning here;
every brain in :mod:`broomworm.candidates` is compared under this same layer.
"""

from broomworm.layer.follower import GradientFollower
from broomworm.layer.needs import NeedsArbiter
from broomworm.layer.occupancy import FREE, OCCUPIED, UNKNOWN, OccupancyGrid
from broomworm.layer.path import bfs, nearest_unswept, next_waypoint, path_to
from broomworm.layer.planner_layer import PlannerLayer, sector_channels

__all__ = [
    "FREE",
    "OCCUPIED",
    "UNKNOWN",
    "GradientFollower",
    "NeedsArbiter",
    "OccupancyGrid",
    "PlannerLayer",
    "bfs",
    "nearest_unswept",
    "next_waypoint",
    "path_to",
    "sector_channels",
]
