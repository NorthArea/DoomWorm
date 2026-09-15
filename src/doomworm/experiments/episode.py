"""Rendering and printing of episode traces. The loop itself lives in ``doomworm.episode``."""

from __future__ import annotations

from pathlib import Path

from doomworm.environments.simple_2d import World
from doomworm.episode import Record, run_episode

__all__ = ["Record", "print_trace", "render_ascii", "run_episode", "save_log", "save_plot"]


def save_log(trace: list[Record], path: Path, top: int = 10) -> None:
    """Write one JSON object per tick (Plan §15 log): pose, sensors, wheels, reward, neurons."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in trace:
            active = sorted((rec.activity or {}).items(), key=lambda kv: -kv[1])[:top]
            row = {
                "tick": rec.tick,
                "x": round(rec.x, 4),
                "y": round(rec.y, 4),
                "heading": round(rec.heading, 4),
                "obstacle": [round(v, 4) for v in rec.obstacle],
                "food": [round(v, 4) for v in rec.food],
                "hunger": round(rec.hunger, 4),
                "motors": [round(v, 4) for v in rec.motors],
                "reward": round(rec.reward, 4),
                "collided": rec.collided,
                "ate": rec.ate,
                "starved": rec.starved,
                "active_neurons": {n: round(v, 4) for n, v in active if v > 0.0},
            }
            f.write(json.dumps(row) + "\n")


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
            if any(
                (wx - o.x) ** 2 + (wy - o.y) ** 2 <= o.radius**2 for o in world.obstacles
            ) or any(w.x <= wx <= w.x1 and w.y <= wy <= w.y1 for w in world.walls):
                grid[r][c] = "#"
    for rec in trace:
        r, c = cell(rec.x, rec.y)
        grid[r][c] = "."
    for f in world.foods:
        r, c = cell(f.x, f.y)
        grid[r][c] = "F"
    if world.target is not None:
        r, c = cell(world.target.x, world.target.y)
        grid[r][c] = "X"
    for d in world.dangers:
        for r in range(rows):
            for c in range(cols):
                wx, wy = (c + 0.5) / sx, (rows - r - 0.5) / sy
                if (wx - d.x) ** 2 + (wy - d.y) ** 2 <= d.radius**2 and grid[r][c] == " ":
                    grid[r][c] = "!"
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
    from matplotlib.patches import Circle, Rectangle

    fig, ax = plt.subplots(figsize=(6, 6))
    for o in world.obstacles:
        ax.add_patch(Circle((o.x, o.y), o.radius, color="tab:red", alpha=0.6))
    for w in world.walls:
        ax.add_patch(Rectangle((w.x, w.y), w.w, w.h, color="tab:gray"))
    for f in world.foods:
        ax.add_patch(Circle((f.x, f.y), f.radius, color="tab:green"))
    if world.target is not None:
        ax.plot(world.target.x, world.target.y, "X", color="tab:orange", ms=12, label="target")
    for d in world.dangers:
        ax.add_patch(Circle((d.x, d.y), d.radius, color="tab:purple", alpha=0.3, hatch="//"))
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
        event += "GOAL" if rec.reached else ""
        event += "DMG" if rec.damaged else ""
        event += "DEAD" if rec.dead and not rec.starved else ""
        event += "DEAD" if rec.starved else ""
        print(
            f"{rec.tick:>4} {rec.x:>6.2f} {rec.y:>6.2f} {rec.heading:>6.2f} | "
            f"{o_l:>4.2f} {o_f:>4.2f} {o_r:>4.2f} | {f_l:>4.2f} {f_f:>4.2f} {f_r:>4.2f} | "
            f"{rec.hunger:>4.2f} | {m_l:>3.0f} {m_r:>3.0f} | {rec.reward:>6.2f} | {event}"
        )
