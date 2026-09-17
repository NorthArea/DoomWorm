"""Views of connectome activity: ASCII propagation tree, raster and active subgraph."""

from __future__ import annotations

from pathlib import Path

from broomworm.brain import ActivityTrace
from broomworm.connectome import Connectome


def propagation_tree(
    connectome: Connectome,
    trace: ActivityTrace,
    source: str,
    threshold: float = 0.01,
    max_depth: int = 3,
    max_children: int = 4,
) -> str:
    """ASCII tree of downstream neurons in the order activity reached them.

    A child is listed under a parent when it is a direct target of the parent
    and became active strictly after it; children are ranked by peak activity.
    """
    first = trace.first_active(threshold)
    peaks = trace.peak()
    lines: list[str] = []
    seen: set[str] = set()

    def describe(nid: str) -> str:
        return f"{nid} (t={first.get(nid, '-')}, peak={peaks.get(nid, 0.0):.3f})"

    def walk(nid: str, prefix: str, depth: int) -> None:
        seen.add(nid)
        if depth >= max_depth:
            return
        candidates = {
            c.target
            for c in connectome.outgoing(nid)
            if c.target in first and c.target not in seen and first[c.target] > first.get(nid, -1)
        }
        children = sorted(candidates, key=lambda c: -peaks[c])[:max_children]
        for i, child in enumerate(children):
            last = i == len(children) - 1
            lines.append(f"{prefix}{'`-- ' if last else '|-- '}{describe(child)}")
            walk(child, prefix + ("    " if last else "|   "), depth + 1)

    lines.append(describe(source))
    walk(source, "", 0)
    return "\n".join(lines)


def plot_raster(trace: ActivityTrace, path: Path, top: int = 30, title: str = "") -> None:
    """Heatmap of the most active neurons over time."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    stim = list(trace.stimulus)
    names = stim + [n for n, _ in trace.top(top - len(stim))]
    data = np.array([[step[n] for n in names] for step in trace.history]).T

    fig, ax = plt.subplots(figsize=(10, max(4, 0.25 * len(names))))
    im = ax.imshow(data, aspect="auto", cmap="magma", vmin=0.0, vmax=max(1e-6, data.max()))
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("tick")
    ax.axvline(trace.hold - 0.5, color="cyan", lw=1, ls="--", label="stimulus off")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title(title or f"stimulus {trace.stimulus}")
    fig.colorbar(im, ax=ax, label="activity")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_active_subgraph(
    connectome: Connectome,
    trace: ActivityTrace,
    path: Path,
    max_nodes: int = 40,
    title: str = "",
) -> None:
    """Graph of the most responsive neurons, node colour = peak activity."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    peaks = trace.peak()
    names = list(trace.stimulus) + [n for n, _ in trace.top(max_nodes - len(trace.stimulus))]
    keep = set(names)
    g = nx.DiGraph()
    g.add_nodes_from(names)
    for nid in names:
        for c in connectome.outgoing(nid):
            if c.target in keep and c.target != nid:
                g.add_edge(nid, c.target)

    pos = nx.spring_layout(g, seed=0, k=0.9)
    responders = [n for n in names if n not in trace.stimulus]
    scale = max(1e-6, max((peaks[m] for m in responders), default=0.0))
    colors = [1.0 if n in trace.stimulus else peaks[n] / scale for n in names]
    fig, ax = plt.subplots(figsize=(9, 9))
    nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.25, arrows=True, arrowsize=8)
    nx.draw_networkx_nodes(
        g, pos, nodelist=names, node_color=colors, cmap="magma", vmin=0, vmax=1, node_size=350
    )
    nx.draw_networkx_labels(g, pos, font_size=7, ax=ax)
    ax.set_title(title or f"neurons responding to {trace.stimulus}")
    ax.set_axis_off()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
