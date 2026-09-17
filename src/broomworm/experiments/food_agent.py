"""Stage 2 demo: hunger motivates the stage-1 agent to seek food.

Adds to the stage-1 brain (no learning):

    hunger  --> HUNGRY            fires once hunger >= HUNGER_THRESHOLD
    HUNGRY  --(+0.9)--> F_LEFT / F_FRONT / F_RIGHT
    food_*  --(+1.0)--> F_*       so F_* fires when hungry and food signal
                                  >= 0.1 (within ~10 units); when sated
                                  only touching food (signal 1.0) fires it.
    F_LEFT  --(-1)--> M_LEFT      food on the left  -> turn left
    F_RIGHT --(-1)--> M_RIGHT     food on the right -> turn right
    F_FRONT --(+1)--> M_LEFT, M_RIGHT   food ahead -> keep driving even if
                                        a side food sensor also fires

Obstacle wiring is unchanged, so the sum of currents decides conflicts.

Run: ``uv run python -m broomworm.experiments.food_agent [--steps N] [--plot]``
"""

from __future__ import annotations

import argparse
from pathlib import Path

from broomworm.adapters import MotorAdapter, SensoryAdapter
from broomworm.brain import Network, Neuron, Simulator, Synapse
from broomworm.environments.simple_2d import AgentState, Food, Obstacle, World
from broomworm.experiments.episode import print_trace, render_ascii, run_episode, save_plot
from broomworm.experiments.obstacle_agent import MOTOR_NEURONS, build_brain, sensor_channels
from broomworm.learning import RewardTracker

HUNGER_THRESHOLD = 0.3
FOOD_NEURONS = {"food_left": "F_LEFT", "food_front": "F_FRONT", "food_right": "F_RIGHT"}
HUNGER_GATE = 0.9


def extend_brain(net: Network) -> Network:
    """Add hunger gating and food attraction to the stage-1 brain in place."""
    net.add_neuron(Neuron("HUNGRY", threshold=HUNGER_THRESHOLD, decay=1.0))
    for nid in FOOD_NEURONS.values():
        net.add_neuron(Neuron(nid, threshold=1.0, decay=1.0))
        net.add_synapse(Synapse("HUNGRY", nid, weight=HUNGER_GATE))
    net.add_synapse(Synapse("F_LEFT", "M_LEFT", weight=-1.0))
    net.add_synapse(Synapse("F_RIGHT", "M_RIGHT", weight=-1.0))
    net.add_synapse(Synapse("F_FRONT", "M_LEFT", weight=1.0))
    net.add_synapse(Synapse("F_FRONT", "M_RIGHT", weight=1.0))
    return net


def build_world() -> World:
    """Stage-1 layout plus two food items; the first is off to the left."""
    return World(
        width=20.0,
        height=20.0,
        agent=AgentState(x=3.0, y=10.0, heading=0.0),
        obstacles=[Obstacle(x=10.0, y=10.0, radius=1.5)],
        foods=[Food(x=6.0, y=16.0), Food(x=16.0, y=5.0)],
        hunger_rate=0.004,
    )


def build_scenario() -> tuple[World, Simulator, SensoryAdapter, MotorAdapter]:
    """Everything needed for :func:`broomworm.experiments.episode.run_episode`."""
    channels = sensor_channels()
    channels.update({ch: (nid, 1.0) for ch, nid in FOOD_NEURONS.items()})
    channels["hunger"] = ("HUNGRY", 1.0)
    sensory = SensoryAdapter(channels=channels, tonic=dict.fromkeys(MOTOR_NEURONS, 1.0))
    motor = MotorAdapter(left="M_LEFT", right="M_RIGHT")
    return build_world(), Simulator(extend_brain(build_brain())), sensory, motor


def main(argv: list[str] | None = None) -> int:
    """Run the demo and print a trace table plus an ASCII map."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--every", type=int, default=25, help="print every N ticks")
    parser.add_argument("--plot", action="store_true", help="save runs/stage2_food.png")
    args = parser.parse_args(argv)

    world, sim, sensory, motor = build_scenario()
    tracker = RewardTracker()
    trace = run_episode(world, sim, sensory, motor, args.steps, reward=tracker)
    print_trace(trace, args.every)
    print()
    print(render_ascii(world, trace))
    print(
        f"\ncollisions: {world.collisions}  food eaten: {world.food_eaten}  "
        f"reward: {tracker.total:.1f} {tracker.breakdown}"
    )

    if args.plot:
        out = Path("runs") / "stage2_food.png"
        title = f"Stage 2: {len(trace)} ticks, ate {world.food_eaten}, {world.collisions} hits"
        save_plot(world, trace, out, title)
        print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
