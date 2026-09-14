"""Neurotransmitter identities used to initialise synapse signs (Plan §11).

Classic GABA-synthesising (unc-25/GAD positive) neurons of the hermaphrodite:
McIntire et al. 1993 (Nature 364:337); confirmed and extended by Gendrel,
Atlas & Hobert 2016 (eLife 5:e17686). Only the synthesising set is used;
GABA-uptake-only neurons reported in 2016 are not treated as inhibitory.
"""

from __future__ import annotations

_D_CLASS = [f"DD{i:02d}" for i in range(1, 7)] + [f"VD{i:02d}" for i in range(1, 14)]
_HEAD_AND_TAIL = ["RMED", "RMEL", "RMER", "RMEV", "RIS", "AVL", "DVB"]

GABA_NEURONS: frozenset[str] = frozenset([*_D_CLASS, *_HEAD_AND_TAIL])
