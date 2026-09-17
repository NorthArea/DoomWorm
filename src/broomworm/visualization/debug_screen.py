"""Debug screen (Plan §41): environment | brain activity, with sensors, motors, reward.

Frames are rendered from an episode trace recorded with ``record_activity=True``
and saved as PNG or GIF; there is no live window.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from broomworm.environments.simple_2d import World
from broomworm.episode import Record


def _neuron_order(trace: list[Record], top: int) -> list[str]:
    peaks: dict[str, float] = {}
    for rec in trace:
        for nid, value in (rec.activity or {}).items():
            peaks[nid] = max(peaks.get(nid, 0.0), value)
    return sorted(peaks, key=lambda n: -peaks[n])[:top]


def draw_frame(fig: Any, world: World, trace: list[Record], tick: int, neurons: list[str]) -> None:
    """Draw one frame into ``fig`` (cleared first)."""
    from matplotlib.patches import Circle, Rectangle, RegularPolygon

    fig.clf()
    rec = trace[tick]
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1.3])
    env = fig.add_subplot(gs[0, 0])
    brain = fig.add_subplot(gs[0, 1])
    text = fig.add_subplot(gs[1, :])

    for o in world.obstacles:
        env.add_patch(Circle((o.x, o.y), o.radius, color="tab:red", alpha=0.6))
    for w in world.walls:
        env.add_patch(Rectangle((w.x, w.y), w.w, w.h, color="tab:gray"))
    for fx, fy in rec.foods:
        env.add_patch(Circle((fx, fy), 0.3, color="tab:green"))
    path = trace[: tick + 1]
    env.plot([r.x for r in path], [r.y for r in path], "-", color="tab:blue", lw=1)
    orientation = rec.heading - math.pi / 2
    agent = RegularPolygon((rec.x, rec.y), 3, radius=world.agent_radius, orientation=orientation)
    agent.set_color("k")
    env.add_patch(agent)
    env.set_xlim(0, world.width)
    env.set_ylim(0, world.height)
    env.set_aspect("equal")
    env.set_title(f"environment  tick {rec.tick}")

    values = [(rec.activity or {}).get(n, 0.0) for n in neurons]
    brain.barh(range(len(neurons)), values, color="tab:purple")
    brain.set_yticks(range(len(neurons)))
    brain.set_yticklabels(neurons, fontsize=7)
    brain.invert_yaxis()
    brain.set_xlim(0, 1)
    brain.set_title("brain activity")

    o_l, o_f, o_r = rec.obstacle
    f_l, f_f, f_r = rec.food
    m_l, m_r = rec.motors
    total = sum(r.reward for r in path)
    flags = (("   COLLISION", rec.collided), ("   ATE", rec.ate), ("   STARVED", rec.starved))
    events = "".join(flag for flag, on in flags if on)
    text.set_axis_off()
    text.text(
        0.0,
        0.9,
        f"Sensors  obstacle L/F/R: {o_l:.2f} {o_f:.2f} {o_r:.2f}    "
        f"food L/F/R: {f_l:.2f} {f_f:.2f} {f_r:.2f}    hunger: {rec.hunger:.2f}\n"
        f"Motors   left: {m_l:.2f}   right: {m_r:.2f}\n"
        f"Reward   tick: {rec.reward:+.2f}   total: {total:+.2f}{events}",
        va="top",
        family="monospace",
        fontsize=9,
        transform=text.transAxes,
    )


def save_debug_frame(
    world: World, trace: list[Record], tick: int, path: Path, top: int = 20
) -> None:
    """Render a single tick to PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 7))
    draw_frame(fig, world, trace, tick, _neuron_order(trace, top))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def save_debug_gif(
    world: World, trace: list[Record], path: Path, every: int = 5, fps: int = 8, top: int = 20
) -> int:
    """Render every ``every``-th tick to an animated GIF; return the frame count."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter

    neurons = _neuron_order(trace, top)
    fig = plt.figure(figsize=(12, 7))
    writer = PillowWriter(fps=fps)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = 0
    with writer.saving(fig, str(path), dpi=80):
        for tick in range(0, len(trace), every):
            draw_frame(fig, world, trace, tick, neurons)
            writer.grab_frame()
            frames += 1
    plt.close(fig)
    return frames
