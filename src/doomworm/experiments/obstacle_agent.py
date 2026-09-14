"""Stage 1 demo: hand-wired 5-neuron brain drives a 2D agent around one obstacle.

Wiring (no learning):

    S_LEFT  --(-1)--> M_RIGHT     obstacle on the left  -> turn right
    S_RIGHT --(-1)--> M_LEFT      obstacle on the right -> turn left
    S_FRONT --(-1)--> M_RIGHT     obstacle ahead        -> turn right
    S_FRONT --(+1)--> M_LEFT      ...and keep the left wheel on even if
                                  S_RIGHT is also firing (front wins).

Differential drive: only the left wheel turning -> turn right, and vice
versa. Both motor neurons receive a tonic current of 1.0, so with no
obstacle the agent drives straight. A sensor neuron fires when its reading
reaches ``SENSOR_THRESHOLD`` and silences the opposite motor for one tick.

Run: ``uv run python -m doomworm.experiments.obstacle_agent [--steps N] [--plot]``
"""

from __future__ import annotations

import argparse
from pathlib import Path

from doomworm.adapters import MotorAdapter, SensoryAdapter
from doomworm.brain import Network, Neuron, Simulator, Synapse
from doomworm.environments.simple_2d import AgentState, Obstacle, World
from doomworm.experiments.episode import print_trace, render_ascii, run_episode, save_plot
from doomworm.learning import RewardTracker

SENSOR_THRESHOLD = 0.5
SENSOR_NEURONS = {"sensor_left": "S_LEFT", "sensor_front": "S_FRONT", "sensor_right": "S_RIGHT"}
MOTOR_NEURONS = ("M_LEFT", "M_RIGHT")


def build_brain() -> Network:
    """Three obstacle sensor neurons inhibiting two motor neurons."""
    net = Network()
    for nid in SENSOR_NEURONS.values():
        net.add_neuron(Neuron(nid, threshold=SENSOR_THRESHOLD, decay=1.0))
    for nid in MOTOR_NEURONS:
        net.add_neuron(Neuron(nid, threshold=1.0, decay=1.0))
    net.add_synapse(Synapse("S_LEFT", "M_RIGHT", weight=-1.0))
    net.add_synapse(Synapse("S_RIGHT", "M_LEFT", weight=-1.0))
    net.add_synapse(Synapse("S_FRONT", "M_RIGHT", weight=-1.0))
    net.add_synapse(Synapse("S_FRONT", "M_LEFT", weight=1.0))
    return net


def sensor_channels() -> dict[str, tuple[str, float]]:
    """Sensory adapter channels for the obstacle sensors."""
    return {ch: (nid, 1.0) for ch, nid in SENSOR_NEURONS.items()}


def build_world() -> World:
    """Agent on the left, facing east, one obstacle straight ahead."""
    return World(
        width=20.0,
        height=20.0,
        agent=AgentState(x=3.0, y=10.0, heading=0.0),
        obstacles=[Obstacle(x=10.0, y=10.0, radius=1.5)],
    )


def build_scenario() -> tuple[World, Simulator, SensoryAdapter, MotorAdapter]:
    """Everything needed for :func:`doomworm.experiments.episode.run_episode`."""
    sensory = SensoryAdapter(channels=sensor_channels(), tonic=dict.fromkeys(MOTOR_NEURONS, 1.0))
    motor = MotorAdapter(left="M_LEFT", right="M_RIGHT")
    return build_world(), Simulator(build_brain()), sensory, motor


def main(argv: list[str] | None = None) -> int:
    """Run the demo and print a trace table plus an ASCII map."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--every", type=int, default=25, help="print every N ticks")
    parser.add_argument("--plot", action="store_true", help="save runs/stage1_obstacle.png")
    args = parser.parse_args(argv)

    world, sim, sensory, motor = build_scenario()
    tracker = RewardTracker()
    trace = run_episode(world, sim, sensory, motor, args.steps, reward=tracker)
    print_trace(trace, args.every)
    print()
    print(render_ascii(world, trace))
    print(f"\ncollisions: {world.collisions}  reward: {tracker.total:.1f} {tracker.breakdown}")

    if args.plot:
        out = Path("runs") / "stage1_obstacle.png"
        save_plot(world, trace, out, f"Stage 1: {len(trace)} ticks, {world.collisions} collisions")
        print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
