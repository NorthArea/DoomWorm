"""Stage 22: where the worm is already right, and where the search overrules it.

Stage 21 showed that rolling the connectome's own proposals forward beats
taking its first answer. That is one number for a whole episode, and it hides
the interesting part: the brain is not uniformly wrong. In some situations its
first answer is already the one the rollout picks, and in others the search
throws it away every time.

This turns a decision log into that map. For every decision it takes:

    gain       what the search bought: the best branch's score minus the score
               of the branch that started with the brain's own first answer
    agreement  how often that first answer *was* the chosen one
    swing      how far the chosen action sits from it, in intent units

and groups them by a small, declared set of situations read off the channels.
The result is a table that says what this architecture is good at by situation
rather than by one reward number — and, for stage 23, which decisions are worth
teaching back.

The situations are matched in order, first match wins, so each decision counts
once. They are deliberately coarse: a finer split would be fitted to one task,
and this map is meant to be read by both tracks.
"""

from __future__ import annotations

import statistics as st
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from wormlab.learning.search import SearchRow

WALL = 0.5  # a front ray this close counts as a wall in the way
SIGNAL = 1e-6  # a gradient channel above this counts as present


def situation(channels: Mapping[str, float]) -> str:
    """One label per decision, matched in this order.

    `wall ahead` comes first because it changes what every other signal means:
    a target straight ahead with a wall between is not a target ahead.
    """
    if channels.get("sensor_front", 0.0) > WALL:
        return "wall ahead"
    if channels.get("aim", 0.0) > SIGNAL:
        return "monster on the gun line"
    if channels.get("prey_left", 0.0) > SIGNAL or channels.get("prey_right", 0.0) > SIGNAL:
        return "monster off to one side"
    if channels.get("target_front", 0.0) > SIGNAL:
        return "exit ahead"
    if channels.get("target_left", 0.0) > SIGNAL or channels.get("target_right", 0.0) > SIGNAL:
        return "exit to one side"
    return "nothing in particular"


SITUATIONS = (
    "wall ahead",
    "monster on the gun line",
    "monster off to one side",
    "exit ahead",
    "exit to one side",
    "nothing in particular",
)


@dataclass(frozen=True)
class Competence:
    """What the search did to the brain's own answer, in one situation."""

    situation: str
    decisions: int
    agreement: float  # the brain's first answer was the chosen one
    gain: float  # mean score the search added over that first answer
    swing: float  # mean |forward| + |turn| distance between the two

    def row(self) -> str:
        """One markdown row."""
        return (
            f"| {self.situation} | {self.decisions} | {100 * self.agreement:.0f} % | "
            f"{self.gain:+.2f} | {self.swing:.2f} |"
        )


def _swing(row: SearchRow) -> float:
    a, b = row.chosen_drive, row.modal_drive
    return abs(a.forward - b.forward) + abs(a.turn - b.turn)


def competence(rows: Sequence[SearchRow]) -> list[Competence]:
    """Group a decision log by situation. Situations with no decisions are dropped."""
    buckets: dict[str, list[SearchRow]] = {}
    for row in rows:
        buckets.setdefault(situation(row.channels), []).append(row)
    out = []
    for name in SITUATIONS:
        group = buckets.get(name)
        if not group:
            continue
        out.append(
            Competence(
                situation=name,
                decisions=len(group),
                agreement=st.fmean(float(r.chosen == 0) for r in group),
                # candidate 0 is the brain's own first answer; the search kept the best
                gain=st.fmean(r.scores[r.chosen] - r.scores[0] for r in group),
                swing=st.fmean(_swing(r) for r in group),
            )
        )
    return out


def table(rows: Sequence[SearchRow]) -> str:
    """The competence map as markdown."""
    lines = [
        "| situation | decisions | brain already right | what search added | how far it moved |",
        "|---|---|---|---|---|",
    ]
    lines += [c.row() for c in competence(rows)]
    return "\n".join(lines) + "\n"
