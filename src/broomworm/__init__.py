"""The home-robot track: everything only a floor robot needs.

The stable platform lives in :mod:`wormlab` — the connectome, the LIF
simulator, the adapters, the body, the search and the benchmark are shared with
the Doom track and change rarely. What is here changes fast: the link to a
machine over a wire, the engineered layer that holds the map and arbitrates
needs, the Roomba-style floor, and the robustness sweep over a sensor preset's
assumed numbers.

Sibling track: :mod:`doomworm`. What each track learns that the other can use
goes in `docs/knowledge/`, not in either package.
"""

from broomworm import presets as _presets  # noqa: F401  (registers the machines)
from broomworm import tasks  # noqa: F401  -- registers the `clean` task
