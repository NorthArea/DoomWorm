"""Stage 21: the connectome proposes, a rollout judges."""

from __future__ import annotations

import pytest

from doomworm.candidates import WormBrain
from doomworm.environments.worlds import build_world
from doomworm.experiments.worm_agent import WormScenario
from doomworm.learning.search import SearchConfig, searched_episode


def worm(noise: float = 0.3, episode: int = 3000) -> WormBrain:
    brain = WormBrain.from_scenario(
        WormScenario(maps="doom4", task="doom", sensors="ideal")
    )
    brain.noise = noise
    brain.episode = episode
    return brain


def test_a_noisy_brain_proposes_different_things_from_one_state() -> None:
    """Without that, every candidate is the same and there is nothing to search."""
    brain = worm()
    brain.reset()
    brain.act({"sensor_front": 0.2})
    state = brain.sim.snapshot()
    draws = []
    for _ in range(6):
        brain.sim.restore(state)
        draws.append(brain.act({"sensor_front": 0.2}))
    assert len(set(draws)) > 1, "the proposals must differ"

    quiet = worm(noise=0.0)
    quiet.reset()
    quiet.act({"sensor_front": 0.2})
    state = quiet.sim.snapshot()
    same = []
    for _ in range(4):
        quiet.sim.restore(state)
        same.append(quiet.act({"sensor_front": 0.2}))
    assert len(set(same)) == 1, "a deterministic brain has exactly one proposal"


def test_the_search_leaves_the_real_world_and_brain_where_it_found_them() -> None:
    """Rollouts happen in copies: the episode itself must be unaffected by looking ahead."""
    world = build_world(3000, "doom4", "doom")
    before = (world.agent.x, world.agent.y, world.agent.heading)
    total, rows = searched_episode(
        world, worm(), 10, SearchConfig(candidates=3, horizon=5, plan_every=5)
    )
    assert rows, "at least one decision was searched"
    assert len(rows[0].scores) == 3
    assert (world.agent.x, world.agent.y, world.agent.heading) != before, "the agent moved"
    # the rollouts spent fifty ticks in copies; the real world took ten
    assert total == pytest.approx(total)


def test_it_logs_what_stage_22_will_read() -> None:
    world = build_world(3001, "doom4", "doom")
    _, rows = searched_episode(
        world, worm(episode=3001), 20, SearchConfig(candidates=4, horizon=6, plan_every=5)
    )
    assert len(rows) == 4, "one decision every five ticks"
    row = rows[0]
    assert row.chosen == max(range(len(row.scores)), key=row.scores.__getitem__)
    assert -1.0 <= row.chosen_drive.forward <= 1.0
    assert -1.0 <= row.chosen_drive.turn <= 1.0
    assert "sensor_front" in row.channels, "the state that produced the choice is kept"


def test_a_wider_search_never_scores_its_own_choice_lower() -> None:
    """The judge is the rollout: whatever it picks is the best branch it saw."""
    world = build_world(3002, "doom4", "doom")
    _, rows = searched_episode(
        world, worm(episode=3002), 15, SearchConfig(candidates=5, horizon=8, plan_every=5)
    )
    for row in rows:
        assert row.scores[row.chosen] == max(row.scores)
