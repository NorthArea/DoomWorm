"""Stage 14: danger zones (Plan §20)."""

import math

import pytest

from wormlab.connectome import default_sensory_mapping
from wormlab.environments.maze import MapConfig, random_world
from wormlab.environments.simple_2d import AgentState, Danger, World
from wormlab.episode import run_episode
from wormlab.experiments import food_agent
from wormlab.experiments.worm_agent import WormScenario
from wormlab.learning import RewardConfig, RewardTracker


def make_world(**kwargs: object) -> World:
    defaults: dict[str, object] = {"agent": AgentState(x=10.0, y=10.0), "speed": 0.5}
    defaults.update(kwargs)
    return World(**defaults)  # type: ignore[arg-type]


def test_danger_is_sensed_like_food() -> None:
    w = make_world(dangers=[Danger(x=14.0, y=10.0)])
    obs = w.observe()
    assert obs.danger_front == pytest.approx(0.25)
    assert (obs.danger_left, obs.danger_right) == (0.0, 0.0)
    assert obs.health == 1.0


def test_danger_is_not_solid_and_costs_health() -> None:
    w = make_world(dangers=[Danger(x=12.0, y=10.0, radius=1.0)], damage_rate=0.25)
    obs = w.step(1.0, 1.0)  # x=10.5, still outside (1.5 > 1.0 + 0.5)
    assert not obs.damaged
    for _ in range(3):
        obs = w.step(1.0, 1.0)
    assert obs.damaged, "moved into the zone without a collision"
    assert w.collisions == 0
    assert w.health == pytest.approx(0.25), "three ticks inside at 0.25 each"
    assert w.damage_taken == 3


def test_health_zero_is_death() -> None:
    w = make_world(dangers=[Danger(x=10.0, y=10.0, radius=2.0)], damage_rate=0.5)
    w.step(0.0, 0.0)
    assert not w.dead
    w.step(0.0, 0.0)
    assert (w.dead, w.starved) == (True, False)


def test_reward_damage_and_death() -> None:
    t = RewardTracker(RewardConfig(new_cell=0.0))
    r1 = t.step(x=0, y=0, ate=False, collided=False, starved=False, damaged=True, dead=False)
    r2 = t.step(x=0, y=0, ate=False, collided=False, starved=False, damaged=True, dead=True)
    r3 = t.step(x=0, y=0, ate=False, collided=False, starved=False, damaged=False, dead=True)
    assert (r1, r2, r3) == (-2.0, -22.0, 0.0)
    assert t.breakdown["death"] == -20.0
    starved = RewardTracker(RewardConfig(new_cell=0.0))
    assert starved.step(x=0, y=0, ate=False, collided=False, starved=True, dead=True) == -20.0
    assert starved.breakdown["death"] == 0.0, "starvation is scored once, not twice"


def test_episode_ends_on_death() -> None:
    world, sim, sensory, motor = food_agent.build_scenario()
    world.dangers = [Danger(x=world.agent.x + 1.2, y=world.agent.y, radius=1.0)]
    world.damage_rate = 0.5
    trace = run_episode(world, sim, sensory, motor, 50, RewardTracker())
    assert trace[-1].dead
    assert len(trace) < 50
    assert sum(1 for r in trace if r.damaged) == 2
    assert trace[-1].danger_signal[1] > 0.0


def test_mapping_routes_danger_to_ash() -> None:
    m = default_sensory_mapping()
    assert {n for n, _ in m.targets("danger_left")} == {"ASHL"}
    assert {n for n, _ in m.targets("danger_front")} == {"ASHL", "ASHR"}


def test_random_world_and_scenario_place_dangers() -> None:
    w = random_world(4, MapConfig(n_dangers=2))
    assert len(w.dangers) == 2
    for d in w.dangers:
        gaps = [math.dist((d.x, d.y), (o.x, o.y)) - d.radius - o.radius for o in w.obstacles]
        assert min(gaps) > 0.0
    sc = WormScenario(task="target", dangers=1, maps="random")
    world = sc.make_world(3)
    assert len(world.dangers) == 1
    assert world.target is not None
    assert sc.params["dangers"] == 1
    d, t = world.dangers[0], world.target
    assert ((d.x - t.x) ** 2 + (d.y - t.y) ** 2) ** 0.5 > d.radius + t.radius
