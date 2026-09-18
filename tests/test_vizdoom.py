"""Track B, stages B4+: the Doom engine behind the World contract (optional ``doom`` group).

The WAD writer is tested without the engine; the engine tests are skipped when
``vizdoom`` is not installed.
"""

import math
import struct
from pathlib import Path

import pytest

from doomworm import DoomguyBrain
from doomworm.levels import doom_world
from doomworm.wad import SCALE, map_lumps, write_wad
from wormlab.episode import run_brain_episode
from wormlab.learning import BenchmarkConfig, RewardTracker, run_benchmark


def test_wad_writer_mirrors_the_layout(tmp_path: Path) -> None:
    layout = doom_world(3, "doom4")
    lumps = dict(map_lumps(layout))
    assert list(dict(map_lumps(layout))) == [
        "MAP01", "THINGS", "LINEDEFS", "SIDEDEFS", "VERTEXES", "SEGS", "SSECTORS",
        "NODES", "SECTORS", "REJECT", "BLOCKMAP",
    ]  # fmt: skip
    things = [
        struct.unpack("<hhhhh", lumps["THINGS"][i : i + 10])
        for i in range(0, len(lumps["THINGS"]), 10)
    ]
    kinds = [t[3] for t in things]
    assert kinds == [1, 2018, 3004], "player start, exit marker, one monster"
    px, py, pangle, _, _ = things[0]
    assert (px, py) == (round(layout.agent.x * SCALE), round(layout.agent.y * SCALE))
    assert pangle == round(math.degrees(layout.agent.heading)) % 360
    n_vertices = len(lumps["VERTEXES"]) // 4
    n_lines = len(lumps["LINEDEFS"]) // 14
    assert n_lines == n_vertices == 4 + 4 * len(layout.walls) + 8 * len(layout.obstacles)
    assert len(lumps["SIDEDEFS"]) // 30 == n_lines
    assert len(lumps["SECTORS"]) == 26
    path = write_wad(layout, tmp_path / "m.wad")
    data = path.read_bytes()
    assert data[:4] == b"PWAD"
    n_lumps, dir_offset = struct.unpack("<ii", data[4:12])
    assert n_lumps == 11
    assert dir_offset + 16 * n_lumps == len(data)


vzd = pytest.importorskip("vizdoom")
from doomworm.engine import vizdoom_world  # noqa: E402


def test_engine_runs_the_same_layout() -> None:
    w = vizdoom_world(3000, "vizdoom4")
    layout = doom_world(3000, "doom4")
    try:
        assert w.walls == layout.walls
        assert w.target == layout.target
        assert math.dist((w.agent.x, w.agent.y), (layout.agent.x, layout.agent.y)) < 0.1
        assert len(w.enemies) == 1
        assert w.observe().ammo == 1.0
        assert not w.finished
        for _ in range(30):
            obs = w.step(1.0, 1.0)
        assert math.dist((w.agent.x, w.agent.y), (layout.agent.x, layout.agent.y)) > 1.0
        assert obs.hunger == 0.0
        for _ in range(20):
            w.step(0.0, 0.0)  # let the momentum die out (the engine coasts about 2 units)
        x0, y0, h0 = w.agent.x, w.agent.y, w.agent.heading
        for _ in range(10):
            w.step(-1.0, 1.0)  # turn left in place
        assert w.agent.heading > h0
        assert math.dist((w.agent.x, w.agent.y), (x0, y0)) < 0.3
    finally:
        w.close()


def test_engine_shots_and_the_exit() -> None:
    w = vizdoom_world(3001, "vizdoom4")
    try:
        shots = 0
        for _ in range(60):
            obs = w.step(0.0, 0.0, fire=True)
            shots += int(obs.fired)
        assert shots >= 3, "the pistol fires about every 14 tics"
        assert w.shots == shots
        assert w.ammo < 50
    finally:
        w.close()
    room = vizdoom_world(3000, "vizdoom1")
    try:
        assert not room.fire_enabled
        assert room.observe().ammo == 0.0
        d = DoomguyBrain()
        trace = run_brain_episode(room, d, 1500, RewardTracker())
        assert room.exited, "the floor walks to the exit marker in the empty Doom room"
        assert trace[-1].reached
        assert room.finished
    finally:
        room.close()


def test_benchmark_runs_on_the_engine() -> None:
    cfg = BenchmarkConfig(
        maps="vizdoom1", task="doom", sensors="ideal", test_seeds=(3000, 3001), steps=300, repeats=1
    )
    res = run_benchmark(DoomguyBrain(), "doomguy", cfg)
    assert len(res.rows) == 2
    assert res.mean("ticks") <= 300
