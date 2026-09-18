"""Stage 3 acceptance: reward separates good behaviour from bad (Plan §9)."""

from wormlab.experiments.reward_demo import evaluate_all


def test_reward_ranks_brains() -> None:
    results = evaluate_all(steps=600)
    by_name = {r.name: r for r in results}
    assert by_name["food"].total > by_name["obstacle"].total > by_name["blind"].total
    assert by_name["blind"].breakdown["collision"] < 0.0
    assert by_name["food"].breakdown["food"] > 0.0


def test_evaluation_is_reproducible() -> None:
    a = evaluate_all(steps=200)
    b = evaluate_all(steps=200)
    assert [(r.name, r.total) for r in a] == [(r.name, r.total) for r in b]
