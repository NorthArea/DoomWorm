"""Stage 6: stimulate a neuron and watch activity spread (Plan §12)."""

import pytest

from wormlab.brain import stimulate
from wormlab.connectome import Connectome, build_network, load_cook2019


@pytest.fixture(scope="module")
def worm() -> Connectome:
    return load_cook2019()


def test_silent_without_stimulus(worm: Connectome) -> None:
    trace = stimulate(build_network(worm), {}, hold=5, settle=5)
    assert all(trace.max_activity(t) == 0.0 for t in range(trace.ticks))


def test_ashl_reaches_command_interneurons(worm: Connectome) -> None:
    trace = stimulate(build_network(worm), {"ASHL": 1.0}, hold=10, settle=20)
    first = trace.first_active(0.01)
    assert first["ASHL"] == 0
    assert 0 < first["AVAL"] <= 6, "direct target; diluted by AVA's 65 inputs"
    assert 0 < first["AVAR"] <= 6
    peaks = trace.peak()
    assert peaks["AVAL"] > 0.0
    assert peaks["AIBL"] > peaks["AVAL"], "AIB gets a larger share of ASH output"


def test_activity_spreads_then_fades(worm: Connectome) -> None:
    trace = stimulate(build_network(worm), {"ASHL": 1.0}, hold=10, settle=120)
    active = [sum(1 for v in trace.history[t].values() if v > 0.01) for t in range(trace.ticks)]
    assert active[0] == 1
    assert max(active[:10]) > 10, "spreads beyond direct targets"
    assert trace.max_activity(trace.ticks - 1) < 0.001, "returns to silence after stimulus"
    assert active[trace.ticks - 1] == 0


def test_supercritical_gain_locks_on(worm: Connectome) -> None:
    trace = stimulate(build_network(worm, gain=1.0, decay=0.5), {"ASHL": 1.0}, hold=10, settle=30)
    assert trace.max_activity(trace.ticks - 1) > 0.5, "documented failure mode: all-on state"


def test_trace_helpers(worm: Connectome) -> None:
    trace = stimulate(build_network(worm), {"ASHL": 1.0, "ASHR": 1.0}, hold=3, settle=2)
    assert trace.ticks == 5
    assert trace.stimulus == {"ASHL": 1.0, "ASHR": 1.0}
    top = trace.top(5)
    assert len(top) == 5
    assert all(n not in trace.stimulus for n, _ in top)
    assert [v for _, v in top] == sorted((v for _, v in top), reverse=True)
    assert trace.top(1, exclude_stimulus=False)[0][0] in trace.stimulus
