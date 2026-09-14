"""Turn a :class:`Connectome` into a simulated :class:`Network` (Plan §11).

Rules:

* one graded LIF neuron per biological neuron, same threshold and decay;
* chemical synapse weight sign: negative if the source is GABAergic, else positive;
* electrical connections are always positive (already expanded to both directions);
* magnitude is proportional to the dataset weight and normalised per target so
  that the absolute incoming weights of every neuron sum to ``gain``.

Stability: with graded neurons the steady-state gain per synaptic hop is
``gain / decay``. Below 1 a stimulus spreads and then fades; at or above 1
the almost entirely excitatory network locks into an all-on state. Defaults
``gain=0.45, decay=0.5`` give 0.9 per hop (see docs/assumptions.md, stage 6).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection

from doomworm.brain import Network, Neuron, Synapse
from doomworm.connectome.model import ConnectionType, Connectome
from doomworm.connectome.neurotransmitters import GABA_NEURONS


def build_network(
    connectome: Connectome,
    *,
    gain: float = 0.45,
    threshold: float = 1.0,
    decay: float = 0.5,
    graded: bool = True,
    inhibitory: Collection[str] = GABA_NEURONS,
) -> Network:
    """Build the network; topology is the connectome's, weights are the initialisation."""
    net = Network()
    for info in connectome.neurons:
        net.add_neuron(Neuron(info.id, threshold=threshold, decay=decay, graded=graded))

    incoming_total: defaultdict[str, float] = defaultdict(float)
    for c in connectome.connections:
        incoming_total[c.target] += abs(c.weight)

    for c in connectome.connections:
        inhibits = c.connection_type is ConnectionType.CHEMICAL and c.source in inhibitory
        sign = -1.0 if inhibits else 1.0
        weight = sign * gain * c.weight / incoming_total[c.target]
        net.add_synapse(Synapse(c.source, c.target, weight=weight, kind=c.connection_type.value))
    return net
