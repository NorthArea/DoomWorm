"""Stage 2 acceptance: hunger motivates the agent to reach food (Plan §8)."""

from itertools import pairwise

from broomworm.experiments.episode import run_episode
from broomworm.experiments.food_agent import HUNGER_THRESHOLD, build_scenario
from broomworm.experiments.obstacle_agent import SENSOR_THRESHOLD


def test_hungry_agent_finds_food() -> None:
    world, sim, sensory, motor = build_scenario()
    trace = run_episode(world, sim, sensory, motor, steps=600)
    assert world.food_eaten >= 1
    eaten = [r for r in trace if r.ate]
    assert all(r.hunger == 0.0 for r in eaten), "hunger resets on eating"


def test_sated_agent_ignores_food() -> None:
    world, sim, sensory, motor = build_scenario()
    trace = run_episode(world, sim, sensory, motor, steps=600)
    # Motors at tick t reflect sensor readings from ticks t-2 and t-1 (two neuron hops),
    # and HUNGRY -> F_* -> M_* takes three hops to switch off after a meal.
    margin = 3 * world.hunger_rate
    since_meal = 10**9
    for i, r in enumerate(trace):
        since_meal = 0 if r.ate else since_meal + 1
        recent_obstacle = max(max(p.obstacle) for p in trace[max(0, i - 2) : i + 1])
        sated = r.hunger < HUNGER_THRESHOLD - margin and since_meal > 3
        if sated and recent_obstacle < SENSOR_THRESHOLD:
            assert r.motors == (1.0, 1.0), f"tick {r.tick}: turned while sated and clear"


def test_hunger_grows_between_meals() -> None:
    world, sim, sensory, motor = build_scenario()
    trace = run_episode(world, sim, sensory, motor, steps=600)
    for prev, cur in pairwise(trace):
        if not cur.ate:
            assert cur.hunger >= prev.hunger
