"""The engineered layer between the sensors and the brain (Plan §3.2).

Map (occupancy grid), path and coverage planning, needs arbitration, the dock
autopilot / bumper reflex / marker search wrapper (:class:`PlannerLayer`) and
the scripted gradient follower that drives the autopilot. No learning here;
every brain in :mod:`wormlab.candidates` is compared under this same layer.
"""

from broomworm.layer.follower import GradientFollower
from broomworm.layer.needs import NeedsArbiter
from broomworm.layer.occupancy import FREE, OCCUPIED, UNKNOWN, OccupancyGrid
from broomworm.layer.path import bfs, nearest_unswept, next_waypoint, path_to
from broomworm.layer.planner_layer import PlannerLayer, sector_channels
from wormlab.environments.sensors import PRESETS

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


def _register() -> None:
    """Announce this track's layer to the shared harness and to PPO.

    ``coverage`` and ``needs`` are conditions a candidate is trained and
    measured under, so the harness has to know them by name (``--layer``,
    ``--planner``). The machinery is the platform's, the decision to switch it
    on is the track's, as with the sensor presets and the ``clean`` task.
    """
    from broomworm.gym_layer import PlannerWrapper  # here: it imports this package back
    from wormlab.learning.bakeoff import register_layer
    from wormlab.learning.rl import register_env_layer

    for mode in ("coverage", "needs"):
        register_layer(
            mode,
            lambda brain, spec, mode=mode: PlannerLayer(  # type: ignore[misc]
                brain, PRESETS[spec.sensors], mode=mode
            ),
        )
        register_env_layer(mode, lambda env, mode=mode: PlannerWrapper(env, mode=mode))  # type: ignore[misc]


_register()
