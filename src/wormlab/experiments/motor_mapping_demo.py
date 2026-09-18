"""Stage 8 demo: what each sensory channel does to the wheels through the connectome.

For every channel a unit signal is applied for ``brain_steps`` ticks, the
activity is averaged over the window (Plan §3.1) and read by the motor
adapter (Plan §14). Untrained weights, so expect small numbers and no
left/right steering: the point is that the sign of touch is "reverse".

Run: ``uv run python -m wormlab.experiments.motor_mapping_demo [--brain-steps K]``
"""

from __future__ import annotations

import argparse

from wormlab.adapters import GroupMotorAdapter
from wormlab.brain import Simulator
from wormlab.connectome import (
    build_network,
    default_motor_mapping,
    default_sensory_mapping,
    load_cook2019,
)
from wormlab.episode import average_activity


def main(argv: list[str] | None = None) -> int:
    """Print motor group means and wheel commands per channel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain-steps", type=int, default=5)
    parser.add_argument("--gain", type=float, default=10.0)
    args = parser.parse_args(argv)

    worm = load_cook2019()
    sensory_map = default_sensory_mapping()
    motor_map = default_motor_mapping()
    motor_map.validate(worm)
    sensory = sensory_map.to_adapter()
    motor = GroupMotorAdapter(
        motor_map.forward,
        motor_map.reversal,
        motor_map.turn_left,
        motor_map.turn_right,
        gain_drive=args.gain,
        gain_turn=args.gain,
    )

    print("motor mapping (Plan §14):")
    print(motor_map.describe())
    print(f"\nunit signal per channel, {args.brain_steps} brain ticks, gain {args.gain}:")
    print(
        f"{'channel':<13}{'fwd':>7}{'rev':>7}{'turnL':>7}{'turnR':>7} | {'drive':>6}{'turn':>7} |"
        f" {'wheel L':>8}{'wheel R':>8}"
    )
    for channel in sensory_map.channels():
        sim = Simulator(build_network(worm))
        currents = sensory({channel: 1.0})
        activity = average_activity([sim.step(currents) for _ in range(args.brain_steps)])
        c = motor.components(activity)
        left, right = motor(activity)
        groups = "".join(
            f"{c[k]:>7.3f}" for k in ("forward", "reversal", "turn_left", "turn_right")
        )
        wheels = f"{left:>8.2f}{right:>8.2f}"
        print(f"{channel:<13}{groups} | {c['drive']:>6.2f}{c['turn']:>7.3f} | {wheels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
