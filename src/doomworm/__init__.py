"""The Doom track: everything that only a Doom player needs (Plan §3).

The stable platform lives in :mod:`wormlab` — the connectome, the LIF
simulator, the adapters, the body, the search and the benchmark are shared with
the other track and change rarely. What is here changes fast: the levels, the
engine behind the world contract, the maps the game itself ships, and the
hand-written player that every trained brain is measured against.
"""

from doomworm.doomguy import DoomguyBrain
from doomworm.levels import LEVELS, doom_world, is_doom_level

__all__ = ["LEVELS", "DoomguyBrain", "doom_world", "is_doom_level"]
