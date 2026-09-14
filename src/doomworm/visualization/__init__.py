"""Debug screen: environment, connectome activity, sensors, motors, reward."""

from doomworm.visualization.brain import plot_active_subgraph, plot_raster, propagation_tree
from doomworm.visualization.debug_screen import save_debug_frame, save_debug_gif

__all__ = [
    "plot_active_subgraph",
    "plot_raster",
    "propagation_tree",
    "save_debug_frame",
    "save_debug_gif",
]
