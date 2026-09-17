"""Stage 3: reward and episode end (Plan §8 «Конец эпизода», §9)."""

import pytest

from broomworm.learning import RewardConfig, RewardTracker


def make_tracker(**kwargs: float) -> RewardTracker:
    return RewardTracker(RewardConfig(**kwargs))


def test_food_reward() -> None:
    t = make_tracker()
    assert t.step(x=0.5, y=0.5, ate=True, collided=False, starved=False) == pytest.approx(
        t.config.food + t.config.new_cell
    )
    assert t.breakdown["food"] == pytest.approx(t.config.food)


def test_collision_penalty_per_tick_and_capped() -> None:
    t = make_tracker(collision=-0.5, max_collision_penalty=1.2, new_cell=0.0)
    ticks = [t.step(x=0.5, y=0.5, ate=False, collided=True, starved=False) for _ in range(4)]
    assert ticks == pytest.approx([-0.5, -0.5, -0.2, 0.0])
    assert t.breakdown["collision"] == pytest.approx(-1.2)


def test_new_cell_bonus_once_per_cell() -> None:
    t = make_tracker(new_cell=0.1, cell_size=1.0)
    first = t.step(x=0.2, y=0.2, ate=False, collided=False, starved=False)
    same = t.step(x=0.9, y=0.9, ate=False, collided=False, starved=False)
    other = t.step(x=1.1, y=0.2, ate=False, collided=False, starved=False)
    assert (first, same, other) == pytest.approx((0.1, 0.0, 0.1))
    assert t.cells_visited == 2


def test_starvation_penalty_once() -> None:
    t = make_tracker(starvation=-20.0, new_cell=0.0)
    assert t.step(x=0.0, y=0.0, ate=False, collided=False, starved=True) == -20.0
    assert t.step(x=0.0, y=0.0, ate=False, collided=False, starved=True) == 0.0


def test_total_is_sum_of_breakdown() -> None:
    t = make_tracker()
    t.step(x=0.0, y=0.0, ate=True, collided=True, starved=False)
    t.step(x=5.0, y=5.0, ate=False, collided=False, starved=True)
    assert t.total == pytest.approx(sum(t.breakdown.values()))
    assert t.total == pytest.approx(10.0 - 0.5 + 0.2 - 20.0)


def test_reward_config_defaults_match_plan() -> None:
    c = RewardConfig()
    assert (c.food, c.collision, c.new_cell, c.starvation) == (10.0, -0.5, 0.1, -20.0)
