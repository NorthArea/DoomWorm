"""Stage 13: target instead of food (Plan §19)."""

import pytest

from broomworm.connectome import default_sensory_mapping
from broomworm.environments.simple_2d import AgentState, Food, Obstacle, Target, World
from broomworm.episode import run_episode
from broomworm.experiments import food_agent
from broomworm.experiments.worm_agent import WormScenario
from broomworm.learning import RewardConfig, RewardTracker


def make_world(**kwargs: object) -> World:
    defaults: dict[str, object] = {"agent": AgentState(x=10.0, y=10.0), "speed": 0.5}
    defaults.update(kwargs)
    return World(**defaults)  # type: ignore[arg-type]


def test_target_is_sensed_like_food() -> None:
    w_t = make_world(target=Target(x=14.0, y=10.0))
    w_f = make_world(foods=[Food(x=14.0, y=10.0)])
    ot, of = w_t.observe(), w_f.observe()
    assert (ot.target_left, ot.target_front, ot.target_right) == (
        of.food_left,
        of.food_front,
        of.food_right,
    )
    assert ot.target_front == pytest.approx(0.25)
    assert (ot.food_left, ot.food_front, ot.food_right) == (0.0, 0.0, 0.0)


def test_reaching_target_counts_and_clears_it() -> None:
    w = make_world(target=Target(x=11.0, y=10.0))
    obs = w.step(1.0, 1.0)
    assert obs.reached
    assert w.targets_reached == 1
    assert w.target is None
    assert w.observe().target_front == 0.0


def test_target_respawns_from_seed() -> None:
    def run(seed: int) -> tuple[float, float]:
        w = make_world(
            target=Target(x=11.0, y=10.0),
            respawn_target=True,
            obstacles=[Obstacle(x=15.0, y=15.0, radius=2.0)],
            seed=seed,
        )
        w.step(1.0, 1.0)
        assert w.target is not None
        return (w.target.x, w.target.y)

    assert run(1) == run(1)
    assert run(1) != run(2)


def test_reward_for_target() -> None:
    t = RewardTracker(RewardConfig(new_cell=0.0))
    assert t.step(x=0, y=0, ate=False, collided=False, starved=False, reached=True) == 10.0
    assert t.breakdown["target"] == 10.0


def test_mapping_routes_target_to_the_food_neurons() -> None:
    m = default_sensory_mapping()
    for side in ("left", "front", "right"):
        assert m.targets(f"target_{side}") == m.targets(f"food_{side}")
    adapter = m.to_adapter()
    assert adapter({"target_left": 0.4}) == adapter({"food_left": 0.4})


def test_episode_records_target_events() -> None:
    world, sim, sensory, motor = food_agent.build_scenario()
    world.foods = []
    world.target = Target(x=world.agent.x + 1.5, y=world.agent.y)
    trace = run_episode(world, sim, sensory, motor, 8, RewardTracker())
    assert not trace[0].reached
    assert trace[0].target is not None
    assert trace[0].target_signal[1] > 0.0
    assert any(r.reached for r in trace)
    assert sum(r.reward for r in trace if r.reached) >= 10.0


def test_worm_scenario_target_task() -> None:
    sc = WormScenario(task="target")
    w = sc.make_world(5)
    assert w.foods == []
    assert w.target is not None
    assert w.respawn_target
    assert sc.params["task"] == "target"
    with pytest.raises(ValueError, match="task"):
        WormScenario(task="nope")
    random_w = WormScenario(task="target", maps="random").make_world(5)
    assert random_w.target is not None
    assert random_w.foods == []
