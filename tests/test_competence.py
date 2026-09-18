"""Stage 22: the map of where the brain is already right."""

from __future__ import annotations

from wormlab.body import Drive
from wormlab.learning.competence import competence, situation, table
from wormlab.learning.search import SearchRow


def row(tick: int, scores: list[float], chosen: int, channels: dict[str, float]) -> SearchRow:
    return SearchRow(
        tick=tick,
        scores=scores,
        chosen=chosen,
        chosen_drive=Drive(forward=1.0, turn=0.5),
        modal_drive=Drive(forward=1.0, turn=0.0),
        channels=channels,
    )


def test_a_wall_ahead_outranks_everything_else() -> None:
    """A target behind a wall is not a target ahead, so the wall is matched first."""
    assert situation({"sensor_front": 0.9, "target_front": 0.5}) == "wall ahead"
    assert situation({"sensor_front": 0.1, "target_front": 0.5}) == "exit ahead"
    assert situation({"aim": 0.7, "target_front": 0.5}) == "monster on the gun line"
    assert situation({"prey_left": 0.2}) == "monster off to one side"
    assert situation({"target_right": 0.3}) == "exit to one side"
    assert situation({}) == "nothing in particular"


def test_it_measures_what_the_search_bought_over_the_brains_own_answer() -> None:
    rows = [
        # the brain's own first answer was kept: no gain, no swing
        row(0, [5.0, 1.0, 2.0], 0, {"target_front": 0.4}),
        # the search found something two points better
        row(5, [1.0, 3.0, 0.0], 1, {"target_front": 0.4}),
    ]
    (entry,) = competence(rows)
    assert entry.situation == "exit ahead"
    assert entry.decisions == 2
    assert entry.agreement == 0.5, "one of the two kept the brain's answer"
    assert entry.gain == 1.0, "mean of 0 and +2"
    assert entry.swing == 0.5, "the drives differ by half a unit of turn"


def test_situations_with_no_decisions_are_left_out() -> None:
    entries = competence([row(0, [1.0], 0, {"sensor_front": 0.9})])
    assert [e.situation for e in entries] == ["wall ahead"]


def test_the_table_is_markdown_with_a_row_per_situation() -> None:
    rows = [
        row(0, [1.0, 2.0], 1, {"sensor_front": 0.9}),
        row(5, [3.0, 1.0], 0, {"aim": 0.8}),
    ]
    out = table(rows).splitlines()
    assert out[0].startswith("| situation |")
    assert len(out) == 4, "header, rule, and one row per situation seen"
    assert "wall ahead" in out[2]
    assert "monster on the gun line" in out[3]
