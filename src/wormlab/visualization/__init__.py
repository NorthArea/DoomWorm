"""Debug screen: environment, connectome activity, sensors, motors, reward."""

from wormlab.visualization.brain import plot_active_subgraph, plot_raster, propagation_tree
from wormlab.visualization.debug_screen import save_debug_frame, save_debug_gif

__all__ = [
    "plot_active_subgraph",
    "plot_raster",
    "propagation_tree",
    "save_debug_frame",
    "save_debug_gif",
]
