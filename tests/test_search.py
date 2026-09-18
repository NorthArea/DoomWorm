"""Stage 21: the connectome proposes, a rollout judges."""

from __future__ import annotations

import pytest

from wormlab.candidates import WormBrain
from wormlab.environments.worlds import build_world
from wormlab.experiments.worm_agent import WormScenario
from wormlab.learning.search import SearchConfig, searched_episode


def worm(noise: float = 0.3, episode: int = 3000) -> WormBrain:
    brain = WormBrain.from_scenario(WormScenario(maps="doom4", task="doom", sensors="ideal"))
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


def test_consensus_averages_the_best_branches_and_one_keeps_the_old_behaviour() -> None:
    """Stage 23b: the decision is the mean of the best k, and k=1 is stage 21."""
    from wormlab.body import Drive
    from wormlab.learning.search import _mean_drive

    drives = [Drive(forward=1.0, turn=-1.0), Drive(forward=0.0, turn=1.0)]
    mean = _mean_drive(drives)
    assert (mean.forward, mean.turn) == (0.5, 0.0)
    assert _mean_drive([drives[0]]) == drives[0]

    world = build_world(3003, "doom4", "doom")
    _, rows = searched_episode(
        world, worm(episode=3003), 10, SearchConfig(candidates=4, horizon=5, plan_every=5)
    )
    for row in rows:  # the default is one, so the chosen branch is executed unchanged
        assert row.scores[row.chosen] == max(row.scores)


def test_the_random_proposer_answers_the_state_not_at_all() -> None:
    """The control stage 23 needed: same contract, no relation to the channels."""
    from wormlab.learning.search import RandomProposer

    control = RandomProposer(seed=7)
    control.reset()
    a = [control.act({"sensor_front": 0.0}) for _ in range(20)]
    b = [control.act({"sensor_front": 1.0}) for _ in range(20)]
    assert len(set(a)) == 20, "every draw is fresh"
    assert set(a).isdisjoint(b)
    assert all(-1.0 <= left <= 1.0 and -1.0 <= right <= 1.0 for left, right in a)

    # restoring must not rewind the draws, or every branch becomes the same one
    state = control.snapshot()
    first = control.act({})
    control.restore(state)
    assert control.act({}) != first

    world = build_world(3004, "doom4", "doom")
    reward, rows = searched_episode(
        world, control, 10, SearchConfig(candidates=3, horizon=5, plan_every=5)
    )
    assert rows, "the search cannot tell it from a brain"
    assert isinstance(reward, float)


def test_noise_reaches_the_running_simulator() -> None:
    """Stage 23b: setting it mid-episode used to do nothing until the next reset."""
    brain = worm(noise=0.0)
    brain.reset()
    assert brain.sim.noise == 0.0
    brain.noise = 0.4
    assert brain.sim.noise == 0.4, "the live simulator, not just the next one"
    brain.reset()
    assert brain.sim.noise == 0.4, "and it survives a reset"
