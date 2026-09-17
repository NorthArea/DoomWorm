"""Stage 7 demo: what each world channel does to the connectome.

Prints the Plan §13 table, then drives every channel with a unit signal and
lists the neurons that respond most, so the biological circuits become visible:
touch -> AVA/AVD (reversal), smell -> AIY/AIA, etc.

Run: ``uv run python -m broomworm.experiments.sensory_mapping_demo [--top N]``
"""

from __future__ import annotations

import argparse

from broomworm.brain import stimulate
from broomworm.connectome import build_network, default_sensory_mapping, load_cook2019


def main(argv: list[str] | None = None) -> int:
    """Print mapping and per-channel responders."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--hold", type=int, default=10)
    args = parser.parse_args(argv)

    worm = load_cook2019()
    mapping = default_sensory_mapping()
    mapping.validate(worm)
    net = build_network(worm)
    adapter = mapping.to_adapter()

    print("sensory mapping (Plan §13):")
    print(mapping.describe())
    print()
    print(f"top {args.top} responders per channel (unit signal for {args.hold} ticks):")
    for channel in mapping.channels():
        stimulus = {n: g for n, g in adapter({channel: 1.0}).items() if g != 0.0}
        trace = stimulate(net, stimulus, hold=args.hold, settle=0)
        first = trace.first_active(0.005)
        responders = ", ".join(
            f"{n}({worm.neuron(n).type[:5]}, t{first.get(n, '-')}, {p:.2f})"
            for n, p in trace.top(args.top)
        )
        print(f"  {channel:<13} {responders}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
