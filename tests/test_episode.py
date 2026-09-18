"""Episode loop: reward wiring and termination (Plan §8, §9)."""

from wormlab.environments.simple_2d import AgentState, World
from wormlab.experiments.episode import run_episode
from wormlab.experiments.food_agent import build_scenario
from wormlab.learning import RewardTracker


def test_episode_ends_on_starvation() -> None:
    world, sim, sensory, motor = build_scenario()
    world.foods = []
    world.hunger_rate = 0.125  # exact in binary: starves at tick 8
    tracker = RewardTracker()
    trace = run_episode(world, sim, sensory, motor, steps=1000, reward=tracker)
    assert len(trace) == 8
    assert trace[-1].starved
    assert tracker.breakdown["starvation"] == tracker.config.starvation


def test_episode_respects_step_limit() -> None:
    world, sim, sensory, motor = build_scenario()
    trace = run_episode(world, sim, sensory, motor, steps=25, reward=RewardTracker())
    assert len(trace) == 25
    assert not trace[-1].starved


def test_trace_reward_matches_tracker() -> None:
    world, sim, sensory, motor = build_scenario()
    tracker = RewardTracker()
    trace = run_episode(world, sim, sensory, motor, steps=300, reward=tracker)
    assert sum(r.reward for r in trace) == tracker.total
    assert any(r.reward >= tracker.config.food for r in trace if r.ate)


def test_without_tracker_reward_is_zero() -> None:
    world = World(agent=AgentState(x=10.0, y=10.0))
    _, sim, sensory, motor = build_scenario()
    trace = run_episode(world, sim, sensory, motor, steps=5)
    assert all(r.reward == 0.0 for r in trace)
