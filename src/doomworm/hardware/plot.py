"""Picture of a drive log (stage 22.2): where the car thought it went, what it saw.

``doomworm plot-log`` draws the odometry path (and the true path when the
log has one), bumper hits, the range readings and the wheel commands over
time. The first real logs are read this way before any number is trusted.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from doomworm.hardware.drive import DriveRow


def plot_drive_log(
    rows: Sequence[DriveRow],
    path: Path | str,
    meta: dict[str, Any] | None = None,
    room: dict[str, Any] | None = None,
) -> Path:
    """Save a PNG with the path and the sensor traces."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_map, ax_rng, ax_whl) = plt.subplots(
        1, 3, figsize=(15, 5), gridspec_kw={"width_ratios": [1.2, 1, 1]}
    )
    ticks = [r.tick for r in rows]
    ox = [r.channels.get("odom_x", 0.0) for r in rows]
    oy = [r.channels.get("odom_y", 0.0) for r in rows]
    ax_map.plot(ox, oy, "-", color="tab:blue", label="odometry")
    if rows and rows[0].truth is not None:
        tx = [r.truth[0] for r in rows if r.truth is not None]
        ty = [r.truth[1] for r in rows if r.truth is not None]
        ax_map.plot(tx, ty, "--", color="tab:green", label="true")
    bumps = [
        r for r in rows if r.channels.get("bumper_left", 0.0) or r.channels.get("bumper_right", 0.0)
    ]
    if bumps:
        ax_map.plot(
            [r.channels.get("odom_x", 0.0) for r in bumps],
            [r.channels.get("odom_y", 0.0) for r in bumps],
            "x",
            color="tab:red",
            label=f"bumper ({len(bumps)})",
        )
    if room is not None:
        ax_map.add_patch(
            Rectangle((0, 0), room["width"], room["height"], fill=False, color="black")
        )
        for x, y, w, h in room.get("walls", []):
            ax_map.add_patch(Rectangle((x, y), w, h, color="lightgray"))
        for x, y, r in room.get("obstacles", []):
            ax_map.add_patch(Circle((x, y), r, color="lightgray"))
        if room.get("marker") is not None:
            ax_map.plot(*room["marker"], "*", color="tab:orange", markersize=12, label="marker")
    if rows:
        x0, y0 = ox[0], oy[0]
        h0 = rows[0].channels.get("odom_heading", 0.0)
        ax_map.annotate(
            "", xy=(x0 + math.cos(h0), y0 + math.sin(h0)), xytext=(x0, y0),
            arrowprops={"arrowstyle": "->", "color": "tab:blue"},
        )  # fmt: skip
    ax_map.set_aspect("equal")
    ax_map.set_title("path (world units)")
    ax_map.legend(loc="best", fontsize=8)

    for key in [k for k in (rows[0].channels if rows else {}) if k.startswith("range_")]:
        ax_rng.plot(ticks, [r.channels[key] for r in rows], label=key, linewidth=0.8)
    for key in ("wall_right", "cliff_left", "cliff_right"):
        if rows and key in rows[0].channels:
            ax_rng.plot(ticks, [r.channels[key] for r in rows], ":", label=key, linewidth=0.8)
    ax_rng.set_ylim(-0.05, 1.05)
    ax_rng.set_title("proximity per ray (1 = touching)")
    ax_rng.set_xlabel("tick")
    ax_rng.legend(fontsize=7)

    ax_whl.plot(ticks, [r.wheels[0] for r in rows], label="left")
    ax_whl.plot(ticks, [r.wheels[1] for r in rows], label="right")
    if rows and "battery" in rows[0].channels:
        ax_whl.plot(ticks, [r.channels["battery"] for r in rows], ":", label="battery")
    ax_whl.set_ylim(-1.1, 1.1)
    ax_whl.set_title("wheel commands")
    ax_whl.set_xlabel("tick")
    ax_whl.legend(fontsize=8)

    title = ", ".join(
        f"{k}={v}" for k, v in (meta or {}).items() if k in ("link", "sensors", "seed")
    )
    fig.suptitle(f"{out.stem}: {len(rows)} ticks  {title}")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
