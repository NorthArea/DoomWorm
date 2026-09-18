"""Stage 6: propagation tree, plots and the §41 debug screen."""

from pathlib import Path

import pytest

from wormlab.brain import stimulate
from wormlab.connectome import Connectome, build_network, load_cook2019
from wormlab.episode import run_episode
from wormlab.experiments import food_agent
from wormlab.learning import RewardTracker
from wormlab.visualization import (
    plot_active_subgraph,
    plot_raster,
    propagation_tree,
    save_debug_frame,
    save_debug_gif,
)


@pytest.fixture(scope="module")
def worm() -> Connectome:
    return load_cook2019()


def test_propagation_tree_lists_downstream_in_order(worm: Connectome) -> None:
    trace = stimulate(build_network(worm), {"ASHL": 1.0}, hold=10, settle=10)
    tree = propagation_tree(worm, trace, "ASHL", threshold=0.01, max_depth=2, max_children=3)
    lines = tree.splitlines()
    assert lines[0].startswith("ASHL (t=0, peak=1.000)")
    assert 1 < len(lines) <= 1 + 3 + 3 * 3
    assert all(line.startswith(("|-- ", "`-- ", "|   ", "    ")) for line in lines[1:])
    direct = {c.target for c in worm.outgoing("ASHL")}
    first_level = [line.split()[1] for line in lines[1:] if line.startswith(("|-- ", "`-- "))]
    assert first_level
    assert set(first_level) <= direct

    wide = propagation_tree(worm, trace, "ASHL", threshold=0.01, max_depth=1, max_children=30)
    assert "AIBL" in wide
    assert "AVAL" in wide


def test_plots_are_written(worm: Connectome, tmp_path: Path) -> None:
    trace = stimulate(build_network(worm), {"ASHL": 1.0}, hold=5, settle=5)
    plot_raster(trace, tmp_path / "r.png", top=10)
    plot_active_subgraph(worm, trace, tmp_path / "g.png", max_nodes=15)
    assert (tmp_path / "r.png").stat().st_size > 1000
    assert (tmp_path / "g.png").stat().st_size > 1000


def test_episode_records_activity_and_foods() -> None:
    world, sim, sensory, motor = food_agent.build_scenario()
    trace = run_episode(world, sim, sensory, motor, 5, record_activity=True)
    assert trace[0].activity is not None
    assert set(trace[0].activity) == set(sim.network.neurons)
    assert len(trace[0].foods) == 2
    plain = run_episode(*food_agent.build_scenario(), 2)
    assert plain[0].activity is None


def test_debug_screen_outputs(tmp_path: Path) -> None:
    world, sim, sensory, motor = food_agent.build_scenario()
    trace = run_episode(world, sim, sensory, motor, 30, RewardTracker(), record_activity=True)
    save_debug_frame(world, trace, 10, tmp_path / "f.png")
    frames = save_debug_gif(world, trace, tmp_path / "d.gif", every=10, fps=5)
    assert frames == 3
    assert (tmp_path / "f.png").stat().st_size > 1000
    assert (tmp_path / "d.gif").read_bytes()[:6] in (b"GIF87a", b"GIF89a")
