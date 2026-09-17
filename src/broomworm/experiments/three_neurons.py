"""Stage 0 demo: INPUT -> N1 -> OUTPUT.

Run: ``uv run python -m broomworm.experiments.three_neurons [input_value]``
"""

from __future__ import annotations

import sys

from broomworm.brain import Network, Neuron, Simulator, Synapse

NEURONS = ("INPUT", "N1", "OUTPUT")


def build_chain() -> Network:
    """Three neurons in a line with unit weights."""
    net = Network()
    for name in NEURONS:
        net.add_neuron(Neuron(name, threshold=1.0, decay=0.5))
    net.add_synapse(Synapse("INPUT", "N1"))
    net.add_synapse(Synapse("N1", "OUTPUT"))
    return net


def main(argv: list[str] | None = None) -> int:
    """Drive the chain with a constant input and print activity per tick."""
    args = sys.argv[1:] if argv is None else argv
    value = float(args[0]) if args else 1.0
    sim = Simulator(build_chain())

    print(f"input = {value}")
    print("tick  " + "  ".join(f"{n:>6}" for n in NEURONS))
    for tick, activity in enumerate(sim.run(8, {"INPUT": value})):
        cells = "  ".join(f"{'*' if activity[n] else '.':>6}" for n in NEURONS)
        print(f"{tick:>4}  {cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
