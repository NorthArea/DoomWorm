"""Closed loop environment -> sensory -> brain -> motor -> environment, plus rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from doomworm.adapters import MotorAdapter, SensoryAdapter
from doomworm.brain import Simulator
from doomworm.environments.simple_2d import World
from doomworm.learning import RewardTracker


@dataclass(frozen=True)
class Record:
    """One tick of an episode."""

    tick: int
    x: float
    y: float
    heading: float
    obstacle: tuple[float, float, float]
    food: tuple[float, float, float]
    hunger: float
    motors: tuple[float, float]
    collided: bool
    ate: bool
    starved: bool = False
    reward: float = 0.0


def run_episode(
    world: World,
    sim: Simulator,
    sensory: SensoryAdapter,
    motor: MotorAdapter,
    steps: int,
    reward: RewardTracker | None = None,
) -> list[Record]:
    """Step the closed loop until ``steps`` ticks or starvation; return the trace.

    ``reward``, when given, scores every tick (Plan §9) and the per-tick value
    lands in :attr:`Record.reward`.
    """
    trace: list[Record] = []
    obs = world.observe()
    for tick in range(steps):
        activity = sim.step(sensory(obs.as_channels()))
        motors = motor(activity)
        obs = world.step(*motors)
        tick_reward = 0.0
        if reward is not None:
            tick_reward = reward.step(
                x=world.agent.x,
                y=world.agent.y,
                ate=obs.ate,
                collided=obs.collided,
                starved=world.starved,
            )
        trace.append(
            Record(
                tick=tick,
                x=world.agent.x,
                y=world.agent.y,
                heading=world.agent.heading,
                obstacle=(obs.sensor_left, obs.sensor_front, obs.sensor_right),
                food=(obs.food_left, obs.food_front, obs.food_right),
                hunger=obs.hunger,
                motors=motors,
                collided=obs.collided,
                ate=obs.ate,
                starved=world.starved,
                reward=tick_reward,
            )
        )
        if world.starved:
            break
    return trace


def render_ascii(world: World, trace: list[Record], cols: int = 60, rows: int = 24) -> str:
    """Top-down map: ``#`` obstacle, ``F`` food, ``.`` path, ``S`` start, ``A`` agent."""
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
    for f in world.foods:
        r, c = cell(f.x, f.y)
        grid[r][c] = "F"
    r, c = cell(trace[0].x, trace[0].y)
    grid[r][c] = "S"
    r, c = cell(trace[-1].x, trace[-1].y)
    grid[r][c] = "A"
    border = "+" + "-" * cols + "+"
    return "\n".join([border, *("|" + "".join(row) + "|" for row in grid), border])


def save_plot(world: World, trace: list[Record], path: Path, title: str) -> None:
    """Save a matplotlib figure of the trajectory."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(6, 6))
    for o in world.obstacles:
        ax.add_patch(Circle((o.x, o.y), o.radius, color="tab:red", alpha=0.6))
    for f in world.foods:
        ax.add_patch(Circle((f.x, f.y), f.radius, color="tab:green"))
    xs, ys = [r.x for r in trace], [r.y for r in trace]
    ax.plot(xs, ys, "-", color="tab:blue", lw=1)
    eaten = [r for r in trace if r.ate]
    if eaten:
        ax.plot([r.x for r in eaten], [r.y for r in eaten], "g*", ms=12, label="ate")
    ax.plot(xs[0], ys[0], "go", label="start")
    ax.plot(xs[-1], ys[-1], "bs", label="end")
    ax.set_xlim(0, world.width)
    ax.set_ylim(0, world.height)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def print_trace(trace: list[Record], every: int) -> None:
    """Print a tick table; always show collision and eating ticks."""
    print(
        "tick      x      y   head |  O_L  O_F  O_R |  F_L  F_F  F_R | hung | M_L M_R |    rew | ev"
    )
    for rec in trace:
        if rec.tick % every and not rec.collided and not rec.ate and not rec.starved:
            continue
        o_l, o_f, o_r = rec.obstacle
        f_l, f_f, f_r = rec.food
        m_l, m_r = rec.motors
        event = ("X" if rec.collided else "") + ("EAT" if rec.ate else "")
        event += "DEAD" if rec.starved else ""
        print(
            f"{rec.tick:>4} {rec.x:>6.2f} {rec.y:>6.2f} {rec.heading:>6.2f} | "
            f"{o_l:>4.2f} {o_f:>4.2f} {o_r:>4.2f} | {f_l:>4.2f} {f_f:>4.2f} {f_r:>4.2f} | "
            f"{rec.hunger:>4.2f} | {m_l:>3.0f} {m_r:>3.0f} | {rec.reward:>6.2f} | {event}"
        )
