"""Stage 16: dirt map, coverage, battery and dock (Plan §20.3)."""

import pytest

from wormlab.connectome import default_sensory_mapping
from wormlab.environments.maze import ApartmentConfig, apartment_world
from wormlab.environments.simple_2d import AgentState, Dock, Wall, World
from wormlab.episode import run_episode
from wormlab.experiments import food_agent
from wormlab.experiments.worm_agent import WormScenario, summarise
from wormlab.learning import RewardConfig, RewardTracker


def test_dirt_map_counts_floor_and_cleans_under_agent() -> None:
    w = World(width=10.0, height=10.0, agent=AgentState(x=5.0, y=5.0), walls=[Wall(0, 0, 10, 2)])
    dirt = w.init_dirt(cell=1.0)
    assert len(dirt.floor) == 10 * 8, "rows 0-1 are wall"
    assert dirt.coverage == 0.0
    cleaned = dirt.clean_around(5.0, 5.0, 1.0)
    assert cleaned == 4, "the four cell centres at distance 0.71"
    assert dirt.clean_around(5.0, 5.0, 1.0) == 0
    assert dirt.coverage == pytest.approx(4 / 80)
    assert (5, 5) not in dirt.dirty


def test_driving_cleans_cells_and_reports_coverage() -> None:
    w = World(agent=AgentState(x=2.0, y=10.0), speed=1.0)
    w.init_dirt()
    obs = w.step(1.0, 1.0)
    assert obs.cleaned >= 1
    for _ in range(5):
        w.step(1.0, 1.0)
    assert 0.0 < w.coverage < 0.1


def test_dock_charges_and_beacon_is_limited() -> None:
    dock = Dock(x=5.0, y=5.0, radius=0.6, beacon_range=3.0, charge_rate=0.1)
    w = World(agent=AgentState(x=5.0, y=5.0), dock=dock, hunger_rate=0.0)
    w.hunger = 0.5
    obs = w.step(0.0, 0.0)
    assert obs.docked
    assert w.battery == pytest.approx(0.6)
    assert obs.battery == pytest.approx(0.6)
    assert w.dockings == 1
    assert w.charging_ticks == 1
    far = World(agent=AgentState(x=15.0, y=5.0), dock=dock)
    assert far.observe().dock_left == far.observe().dock_front == far.observe().dock_right == 0.0
    near = World(agent=AgentState(x=7.0, y=5.0, heading=3.14159), dock=dock)
    assert near.observe().dock_front == pytest.approx(0.5)


def test_docking_counted_once_per_visit() -> None:
    dock = Dock(x=5.0, y=5.0, radius=0.6)
    w = World(agent=AgentState(x=5.0, y=5.0), dock=dock, speed=1.0)
    w.step(0.0, 0.0)
    w.step(0.0, 0.0)
    assert w.dockings == 1
    w.step(1.0, 1.0)  # leaves
    assert not w.docked
    w.step(-1.0, -1.0)  # returns
    assert w.dockings == 2


def test_reward_clean_and_dock() -> None:
    t = RewardTracker(RewardConfig(new_cell=0.0))
    r = t.step(x=0, y=0, ate=False, collided=False, starved=False, cleaned=3)
    assert r == pytest.approx(1.5), "a cell is a fifth of a food item"
    r = t.step(x=0, y=0, ate=False, collided=False, starved=False, docked=True, battery=0.2)
    assert r == 10.0
    r = t.step(x=0, y=0, ate=False, collided=False, starved=False, docked=True, battery=0.3)
    assert r == 0.0, "still docked: no second reward"
    r = t.step(x=0, y=0, ate=False, collided=False, starved=False, docked=False)
    r = t.step(x=0, y=0, ate=False, collided=False, starved=False, docked=True, battery=0.9)
    assert r == 0.0, "docking with a full battery earns nothing"
    assert t.breakdown["dock"] == 10.0


def test_dock_bonus_once_per_discharge_cycle() -> None:
    t = RewardTracker(RewardConfig(new_cell=0.0))

    def tick(docked: bool, battery: float) -> float:
        return t.step(
            x=0, y=0, ate=False, collided=False, starved=False, docked=docked, battery=battery
        )

    assert tick(True, 0.25) == 10.0
    # jitter at the dock edge while still low: leave, re-enter, leave, re-enter
    for _ in range(3):
        tick(False, 0.26)
        assert tick(True, 0.26) == 0.0, "not re-armed"
    # a full charge re-arms the bonus, the next low docking pays again
    tick(True, 0.95)
    tick(False, 0.5)
    assert tick(True, 0.3) == 10.0
    assert t.breakdown["dock"] == 20.0


def test_mapping_routes_dock_to_food_neurons() -> None:
    m = default_sensory_mapping()
    for side in ("left", "front", "right"):
        assert m.targets(f"dock_{side}") == m.targets(f"food_{side}")


def test_clean_task_scenario() -> None:
    sc = WormScenario(task="clean", maps="apartment")
    w = sc.make_world(1)
    assert w.dock is not None
    assert w.dirt is not None
    assert w.foods == []
    assert (w.agent.x, w.agent.y) == (w.dock.x, w.dock.y)
    assert w.hunger_rate == pytest.approx(0.002)
    assert w.coverage == 0.0
    assert sc.params["task"] == "clean"


def test_episode_records_coverage_and_battery() -> None:
    world, sim, sensory, motor = food_agent.build_scenario()
    world.foods = []
    world.init_dirt()
    world.dock = Dock(x=world.agent.x, y=world.agent.y)
    trace = run_episode(world, sim, sensory, motor, 20, RewardTracker())
    assert trace[0].docked
    assert trace[-1].coverage > 0.0
    assert 0.0 < trace[-1].battery <= 1.0
    summary = summarise(trace, world)
    assert summary["coverage"] == pytest.approx(world.coverage)
    assert summary["dockings"] == 1


def test_apartment_dirt_excludes_walls() -> None:
    w = apartment_world(2, ApartmentConfig(furniture_per_room=0))
    dirt = w.init_dirt()
    assert 380 <= len(dirt.floor) <= 400, "thin partitions hide almost no cell centres"
    with_furniture = apartment_world(2, ApartmentConfig(furniture_per_room=1)).init_dirt()
    assert len(with_furniture.floor) < len(dirt.floor)
