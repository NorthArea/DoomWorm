"""Track B, stage B1: mini-Doom without Doom (Plan §21-23, §27-32) on the 2D platform."""

import math
from pathlib import Path

import numpy as np
import pytest

from doomworm.candidates import (
    CandidateSpec,
    DoomguyBrain,
    RNNBrain,
    ScriptedBrain,
    WormBrain,
    build_candidate,
)
from doomworm.cli import main
from doomworm.connectome import default_motor_mapping, load_cook2019
from doomworm.environments.doom.levels import LEVELS, doom_world
from doomworm.environments.gym_env import DoomwormEnv
from doomworm.environments.simple_2d import AgentState, Enemy, Target, Wall, World
from doomworm.environments.worlds import build_world
from doomworm.episode import fire_of, run_brain_episode
from doomworm.experiments.worm_agent import WormScenario
from doomworm.learning import BenchmarkConfig, RewardConfig, RewardTracker, run_benchmark


def arena(**kwargs: object) -> World:
    defaults: dict[str, object] = {
        "agent": AgentState(x=5.0, y=10.0, heading=0.0),
        "fire_enabled": True,
        "hunger_rate": 0.0,
    }
    defaults.update(kwargs)
    return World(**defaults)  # type: ignore[arg-type]


# --- world mechanics -------------------------------------------------------------


def test_enemy_is_sensed_on_the_danger_channels() -> None:
    w = arena(enemies=[Enemy(x=9.0, y=10.0)])
    obs = w.observe()
    assert obs.danger_front == pytest.approx(0.25)
    assert (obs.danger_left, obs.danger_right) == (0.0, 0.0)
    assert obs.ammo == 1.0
    assert arena(fire_enabled=False).observe().ammo == 0.0
    hidden = arena(enemies=[Enemy(x=9.0, y=10.0)], walls=[Wall(x=7.0, y=9.0, w=0.3, h=2.0)])
    assert hidden.observe().danger_front == 0.0, "a monster behind a wall is not seen"
    assert hidden.observe().aim == 0.0


def test_aim_channel_measures_the_offset_from_the_gun_line() -> None:
    assert arena(enemies=[Enemy(x=9.0, y=10.0)]).observe().aim == pytest.approx(1.0)
    off = arena(enemies=[Enemy(x=9.0, y=11.0)])  # 14 degrees off a 30-degree sector
    assert off.observe().aim == pytest.approx(1.0 - math.atan2(1.0, 4.0) / (math.pi / 6))
    assert arena(enemies=[Enemy(x=5.0, y=13.0)]).observe().aim == 0.0, "outside the sector"
    assert arena().observe().aim == 0.0


def test_enemy_in_range_with_line_of_sight_hurts() -> None:
    w = arena(enemies=[Enemy(x=7.0, y=10.0, attack_range=3.0, damage=0.1)])
    obs = w.step(0.0, 0.0)
    assert obs.damaged
    assert w.health == pytest.approx(0.9)
    blocked = arena(
        enemies=[Enemy(x=7.0, y=10.0, attack_range=3.0, damage=0.1)],
        walls=[Wall(x=6.0, y=9.0, w=0.3, h=2.0)],
    )
    assert not blocked.step(0.0, 0.0).damaged, "a wall between them: no line of sight"
    far = arena(enemies=[Enemy(x=9.0, y=10.0, attack_range=3.0)])
    assert not far.step(0.0, 0.0).damaged


def test_fire_hits_the_enemy_in_the_front_cone_and_kills_it() -> None:
    w = arena(enemies=[Enemy(x=9.0, y=10.0, health=2, attack_range=0.0)], fire_cooldown=2)
    obs = w.step(0.0, 0.0, fire=True)
    assert (obs.fired, obs.hit, obs.killed) == (True, 1, 0)
    assert w.enemies[0].health == 1
    obs = w.step(0.0, 0.0, fire=True)
    assert not obs.fired, "cooldown: one shot every 2 ticks"
    obs = w.step(0.0, 0.0, fire=True)
    assert (obs.fired, obs.hit, obs.killed) == (True, 1, 1)
    assert w.enemies == []
    assert (w.shots, w.hits, w.kills) == (2, 2, 1)
    assert w.ammo == w.ammo_max - 2
    assert w.observe().ammo == pytest.approx((w.ammo_max - 2) / w.ammo_max)


def test_fire_misses_off_the_gun_line_beyond_range_or_behind_a_wall() -> None:
    side = arena(enemies=[Enemy(x=5.0, y=13.0, attack_range=0.0)])
    assert side.step(0.0, 0.0, fire=True).hit == 0
    # 4 units ahead, 1 unit off the line: the 0.6-unit body is missed (a hitscan, not a cone)
    off = arena(enemies=[Enemy(x=9.0, y=11.0, attack_range=0.0)])
    assert off.step(0.0, 0.0, fire=True).hit == 0
    grazing = arena(enemies=[Enemy(x=9.0, y=10.5, attack_range=0.0)])
    assert grazing.step(0.0, 0.0, fire=True).hit == 1
    far = arena(enemies=[Enemy(x=15.0, y=10.0, attack_range=0.0)], fire_range=6.0)
    assert far.step(0.0, 0.0, fire=True).hit == 0
    walled = arena(
        enemies=[Enemy(x=9.0, y=10.0, attack_range=0.0)], walls=[Wall(x=7.0, y=9.0, w=0.3, h=2.0)]
    )
    assert walled.step(0.0, 0.0, fire=True).hit == 0
    no_gun = arena(enemies=[Enemy(x=9.0, y=10.0, attack_range=0.0)], fire_enabled=False)
    obs = no_gun.step(0.0, 0.0, fire=True)
    assert not obs.fired
    assert no_gun.shots == 0


def test_the_magazine_is_finite_like_the_doom_pistol() -> None:
    w = arena(enemies=[Enemy(x=9.0, y=10.0, health=999, attack_range=0.0)], ammo=3, fire_cooldown=0)
    for _ in range(3):
        assert w.step(0.0, 0.0, fire=True).fired
    assert w.ammo == 0
    assert not w.fire_enabled
    assert w.observe().ammo == 0.0
    obs = w.step(0.0, 0.0, fire=True)
    assert not obs.fired, "out of ammo: the trigger does nothing"
    assert w.shots == 3


def test_walking_enemy_closes_in_only_when_it_sees_the_agent() -> None:
    w = arena(enemies=[Enemy(x=12.0, y=10.0, speed=0.5, attack_range=1.0, damage=0.0)])
    w.step(0.0, 0.0)
    assert w.enemies[0].x == pytest.approx(11.5)
    hidden = arena(
        enemies=[Enemy(x=12.0, y=10.0, speed=0.5, attack_range=1.0)],
        walls=[Wall(x=8.0, y=8.0, w=0.4, h=4.0)],
    )
    hidden.step(0.0, 0.0)
    assert hidden.enemies[0].x == 12.0, "cannot see the agent: stays"
    for _ in range(30):
        w.step(0.0, 0.0)
    assert w.enemies[0].x >= 5.5, "stops at half its attack range"


def test_exit_ends_the_episode() -> None:
    w = arena(target=Target(x=6.0, y=10.0), exit_ends=True, speed=0.5)
    w.step(1.0, 1.0)
    assert w.exited
    assert w.finished
    assert not w.dead
    assert w.targets_reached == 1
    assert w.target is None


def test_ray_distance_limit_and_line_of_sight() -> None:
    w = arena(walls=[Wall(x=10.0, y=0.0, w=0.5, h=20.0)])
    assert w.ray_distance(0.0) == pytest.approx(4.0), "capped at the sensor range"
    assert w.ray_distance(0.0, limit=100.0) == pytest.approx(5.0)
    assert w.line_of_sight(9.0, 10.0)
    assert not w.line_of_sight(12.0, 10.0)


# --- reward ----------------------------------------------------------------------


def test_reward_hit_kill_exit() -> None:
    t = RewardTracker(RewardConfig(new_cell=0.0))

    def tick(hit: int = 0, killed: int = 0, reached: bool = False) -> float:
        return t.step(
            x=0.0, y=0.0, ate=False, collided=False, starved=False,
            hit=hit, killed=killed, reached=reached,
        )  # fmt: skip

    assert tick(hit=1) == 2.0
    assert tick(hit=1, killed=1) == 12.0
    assert tick(reached=True) == 10.0
    assert t.breakdown["kill"] == 10.0
    assert t.breakdown["hit"] == 4.0


# --- levels ----------------------------------------------------------------------


@pytest.mark.parametrize("level", LEVELS)
def test_levels_are_seeded_and_match_the_plan(level: str) -> None:
    a, b = doom_world(7, level), doom_world(7, level)
    assert (a.agent, a.target) == (b.agent, b.target)
    assert [(e.x, e.y) for e in a.enemies] == [(e.x, e.y) for e in b.enemies]
    other = doom_world(8, level)
    assert other.agent != a.agent
    n = int(level[4:])
    assert a.target is not None
    assert a.exit_ends
    assert a.hunger_rate == 0.0
    assert len(a.enemies) == (0 if n < 3 else 2 if n == 6 else 1)
    assert a.fire_enabled == (n >= 4)
    assert all(e.speed > 0 for e in a.enemies) == (n >= 5) or not a.enemies
    assert math.dist((a.agent.x, a.agent.y), (a.target.x, a.target.y)) >= 5.0
    assert a.clearance(a.target.x, a.target.y) > 0.6
    for e in a.enemies:
        assert a.clearance(e.x, e.y) > e.radius
    if n in (2, 3, 4, 5):
        assert len(a.walls) == 2, "a partition with one doorway"
        assert len(a.obstacles) == 3
    if n == 6:
        assert a.rooms, "apartment rooms"


def test_build_world_dispatches_doom_levels() -> None:
    w = build_world(3, "doom4", "doom")
    assert w.fire_enabled
    assert w.enemies
    with pytest.raises(ValueError, match="task"):
        build_world(3, "doom4", "clean")
    with pytest.raises(ValueError, match="doom level"):
        build_world(3, "random", "doom")


# --- brains ----------------------------------------------------------------------


def test_fire_of_reads_the_optional_attribute() -> None:
    assert not fire_of(ScriptedBrain(lambda _c: (0.0, 0.0)))
    d = DoomguyBrain()
    d.fire = 0.9
    assert fire_of(d)


def test_doomguy_shoots_a_visible_enemy_and_walks_to_the_exit() -> None:
    d = DoomguyBrain()
    d.reset()
    assert d.act({"ammo": 1.0, "danger_front": 0.3, "aim": 0.9}) == (0.0, 0.0)
    assert d.fire == 1.0
    d.act({"ammo": 1.0, "danger_front": 0.3, "aim": 0.3})
    assert d.fire == 0.0, "in the sector but off the line: nudges instead of firing"
    d.act({"ammo": 1.0, "danger_left": 0.3})
    assert d.fire == 0.0, "not facing it yet"
    assert d.act({"target_front": 0.2}) == (1.0, 1.0)
    w = build_world(3001, "doom5", "doom")
    trace = run_brain_episode(w, d, 600, RewardTracker())
    assert w.kills == 1, "the floor kills the walking enemy on level 5 as it comes into range"
    assert w.hits == w.shots, "every shot was taken on the gun line"
    assert w.exited
    assert trace[-1].tick < 599


def test_worm_brain_exposes_a_trigger_from_the_pharyngeal_group() -> None:
    mapping = default_motor_mapping()
    assert mapping.fire == ["M3L", "M3R", "M4", "MCL", "MCR"]
    mapping.validate(load_cook2019())
    sc = WormScenario(maps="doom4", task="doom")
    brain = WormBrain.from_scenario(sc)
    assert brain.trigger is not None
    brain.reset()
    brain.act({"danger_front": 1.0})
    assert 0.0 <= brain.fire <= 1.0
    assert sc.params["gain_fire"] == 10.0
    assert sc.params["maps"] == "doom4"
    with pytest.raises(ValueError, match="maps"):
        WormScenario(maps="doom9")


def test_rnn_gets_a_third_output_on_the_doom_task(tmp_path: Path) -> None:
    two = build_candidate(CandidateSpec("rnn", sensors="ideal"))
    three = build_candidate(CandidateSpec("rnn", maps="doom4", task="doom", sensors="ideal"))
    assert isinstance(two, RNNBrain)
    assert isinstance(three, RNNBrain)
    assert (two.outputs, three.outputs) == (2, 3)
    assert three.n_weights == two.n_weights + two.hidden + 1
    three.reset()
    three.act({"danger_front": 1.0})
    assert -1.0 <= three.fire <= 1.0
    path = tmp_path / "rnn3.json"
    three.save(path)
    loaded = RNNBrain.from_file(path)
    assert loaded.outputs == 3
    assert loaded.get_weights() == pytest.approx(three.get_weights())


def test_saved_worm_brain_reloads_with_a_trigger_and_same_wheels(tmp_path: Path) -> None:
    sc = WormScenario()
    brain = WormBrain.from_scenario(sc)
    path = tmp_path / "w.json"
    brain.save(path)
    loaded = WormBrain.from_file(path, maps="doom3", task="doom")
    assert loaded.trigger is not None
    assert loaded.meta["params"]["task"] == "doom"
    w = loaded.meta["params"]
    assert w["maps"] == "doom3"


# --- platform: gym env and benchmark -----------------------------------------------


def test_gym_env_with_a_trigger() -> None:
    env = DoomwormEnv(
        lambda s: build_world(s, "doom4", "doom"), sensors="ideal", steps=30, fire=True
    )
    assert env.action_space.shape == (3,)
    obs, _ = env.reset(seed=1)
    assert "ammo" in env.channel_names
    assert obs[env.channel_names.index("ammo")] == 1.0
    _, _, _, _, info = env.step(np.array([0.0, 0.0, 1.0]))
    assert info["shots"] == 1
    plain = DoomwormEnv(lambda s: build_world(s, "doom4", "doom"), sensors="ideal", steps=30)
    plain.reset(seed=1)
    _, _, _, _, info = plain.step(np.array([0.0, 0.0]))
    assert info["shots"] == 0


def test_gym_env_terminates_at_the_exit() -> None:
    def factory(seed: int) -> World:
        w = arena(target=Target(x=6.0, y=10.0), exit_ends=True, speed=0.5)
        return w

    env = DoomwormEnv(factory, sensors="ideal", steps=30)
    env.reset(seed=0)
    _, _, terminated, truncated, info = env.step(np.array([1.0, 1.0]))
    assert terminated
    assert not truncated
    assert info["exited"]


def test_benchmark_rows_carry_the_doom_counters(tmp_path: Path) -> None:
    cfg = BenchmarkConfig(
        maps="doom5", task="doom", sensors="ideal", test_seeds=(3001, 3002), steps=400, repeats=1
    )
    res = run_benchmark(DoomguyBrain(), "doomguy", cfg)
    assert {"kills", "hits", "shots", "exited"} <= set(res.summary())
    assert res.mean("shots") > 0
    args = ["benchmark", "--scripted", "doomguy", "--maps", "doom4", "--task", "doom"]
    out = tmp_path / "b"
    extra = ["--sensors", "ideal", "--test-seeds", "1", "--repeats", "1", "--steps", "50"]
    assert main([*args, *extra, "--out-dir", str(out)]) == 0
    table = (out / "leaderboard.md").read_text()
    assert "kills" in table.splitlines()[0]


def test_play_cli_on_a_doom_level(tmp_path: Path) -> None:
    sc = WormScenario()
    path = tmp_path / "w.json"
    WormBrain.from_scenario(sc).save(path)
    args = ["play", "--brain", str(path), "--maps", "doom4", "--task", "doom"]
    assert main([*args, "--seed", "3", "--steps", "20", "--every", "10"]) == 0


def test_ncp_gets_a_third_output_on_the_doom_task(tmp_path: Path) -> None:
    pytest.importorskip("ncps")
    from doomworm.candidates.ncp import NCPBrain

    three = build_candidate(CandidateSpec("ncp", maps="doom4", task="doom", sensors="ideal"))
    assert isinstance(three, NCPBrain)
    assert three.outputs == 3
    three.reset()
    three.act({"danger_front": 1.0})
    assert -1.0 <= three.fire <= 1.0
    path = tmp_path / "ncp3.json"
    three.save(path)
    loaded = NCPBrain.from_file(path)
    assert loaded.outputs == 3
    assert loaded.n_weights == three.n_weights
