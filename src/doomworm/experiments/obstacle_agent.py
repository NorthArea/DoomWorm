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
from dataclasses import dataclass
from pathlib import Path

from doomworm.adapters import MotorAdapter, SensoryAdapter
from doomworm.brain import Network, Neuron, Simulator, Synapse
from doomworm.environments.simple_2d import AgentState, Obstacle, World

SENSOR_THRESHOLD = 0.5
SENSOR_NEURONS = {"sensor_left": "S_LEFT", "sensor_front": "S_FRONT", "sensor_right": "S_RIGHT"}
MOTOR_NEURONS = ("M_LEFT", "M_RIGHT")


@dataclass(frozen=True)
class Record:
    """One tick of an episode."""

    tick: int
    x: float
    y: float
    heading: float
    sensors: tuple[float, float, float]
    motors: tuple[float, float]
    collided: bool


def build_brain() -> Network:
    """Three sensor neurons inhibiting two motor neurons."""
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


def build_world() -> World:
    """Agent on the left, facing east, one obstacle straight ahead."""
    return World(
        width=20.0,
        height=20.0,
        agent=AgentState(x=3.0, y=10.0, heading=0.0),
        obstacles=[Obstacle(x=10.0, y=10.0, radius=1.5)],
    )


def build_scenario() -> tuple[World, Simulator, SensoryAdapter, MotorAdapter]:
    """Everything needed for :func:`run_episode`."""
    sensory = SensoryAdapter(
        channels={ch: (nid, 1.0) for ch, nid in SENSOR_NEURONS.items()},
        tonic=dict.fromkeys(MOTOR_NEURONS, 1.0),
    )
    motor = MotorAdapter(left="M_LEFT", right="M_RIGHT")
    return build_world(), Simulator(build_brain()), sensory, motor


def run_episode(
    world: World,
    sim: Simulator,
    sensory: SensoryAdapter,
    motor: MotorAdapter,
    steps: int,
) -> list[Record]:
    """Close the loop environment -> sensory -> brain -> motor -> environment."""
    trace: list[Record] = []
    obs = world.observe()
    for tick in range(steps):
        activity = sim.step(sensory(obs.as_channels()))
        motors = motor(activity)
        obs = world.step(*motors)
        trace.append(
            Record(
                tick=tick,
                x=world.agent.x,
                y=world.agent.y,
                heading=world.agent.heading,
                sensors=(obs.sensor_left, obs.sensor_front, obs.sensor_right),
                motors=motors,
                collided=obs.collided,
            )
        )
    return trace


def render_ascii(world: World, trace: list[Record], cols: int = 60, rows: int = 24) -> str:
    """Top-down map: ``#`` obstacle, ``.`` path, ``S`` start, ``A`` agent."""
    grid = [[" "] * cols for _ in range(rows)]
    sx, sy = cols / world.width, rows / world.height

    def cell(x: float, y: float) -> tuple[int, int]:
        c = min(cols - 1, max(0, int(x * sx)))
        r = min(rows - 1, max(0, rows - 1 - int(y * sy)))
        return r, c

    for r in range(rows):
        for c in range(cols):
            wx, wy = (c + 0.5) / sx, (rows - r - 0.5) / sy
            if any((wx - o.x) ** 2 + (wy - o.y) ** 2 <= o.radius**2 for o in world.obstacles):
                grid[r][c] = "#"
    for rec in trace:
        r, c = cell(rec.x, rec.y)
        grid[r][c] = "."
    r, c = cell(trace[0].x, trace[0].y)
    grid[r][c] = "S"
    r, c = cell(trace[-1].x, trace[-1].y)
    grid[r][c] = "A"
    border = "+" + "-" * cols + "+"
    return "\n".join([border, *("|" + "".join(row) + "|" for row in grid), border])


def save_plot(world: World, trace: list[Record], path: Path) -> None:
    """Save a matplotlib figure of the trajectory."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(6, 6))
    for o in world.obstacles:
        ax.add_patch(Circle((o.x, o.y), o.radius, color="tab:red", alpha=0.6))
    xs, ys = [r.x for r in trace], [r.y for r in trace]
    ax.plot(xs, ys, "-", color="tab:blue", lw=1)
    ax.plot(xs[0], ys[0], "go", label="start")
    ax.plot(xs[-1], ys[-1], "bs", label="end")
    ax.set_xlim(0, world.width)
    ax.set_ylim(0, world.height)
    ax.set_aspect("equal")
    ax.set_title(f"Stage 1: {len(trace)} ticks, {world.collisions} collisions")
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """Run the demo and print a trace table plus an ASCII map."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--every", type=int, default=10, help="print every N ticks")
    parser.add_argument("--plot", action="store_true", help="save runs/stage1_obstacle.png")
    args = parser.parse_args(argv)

    world, sim, sensory, motor = build_scenario()
    trace = run_episode(world, sim, sensory, motor, args.steps)

    print("tick      x      y   head |  S_L   S_F   S_R | M_L M_R | hit")
    for rec in trace:
        if rec.tick % args.every and not rec.collided:
            continue
        s_l, s_f, s_r = rec.sensors
        m_l, m_r = rec.motors
        print(
            f"{rec.tick:>4} {rec.x:>6.2f} {rec.y:>6.2f} {rec.heading:>6.2f} | "
            f"{s_l:>4.2f}  {s_f:>4.2f}  {s_r:>4.2f} | {m_l:>3.0f} {m_r:>3.0f} | "
            f"{'X' if rec.collided else ''}"
        )
    print()
    print(render_ascii(world, trace))
    print(f"\ncollisions: {world.collisions}")

    if args.plot:
        out = Path("runs") / "stage1_obstacle.png"
        save_plot(world, trace, out)
        print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
