"""Stage 24: the vehicle is not part of the brain's contract (Plan §2.2, §3.3)."""

from __future__ import annotations

import pytest

from wormlab.body import (
    DifferentialDrive,
    Drive,
    MecanumDrive,
    build_body,
    drive_of,
)


def test_intent_is_vehicle_neutral_and_clipped() -> None:
    drive = Drive(forward=0.5, turn=-0.25, strafe=0.1, fire=0.9)
    assert drive.pulls_trigger
    assert not Drive(fire=0.4).pulls_trigger
    assert drive.with_fire(0.0).forward == 0.5
    assert Drive(forward=2.0, turn=-3.0).clipped() == Drive(forward=1.0, turn=-1.0)


def test_two_wheels_round_trip_exactly() -> None:
    """Old brains speak wheels; the differential body must not change a single number."""
    body = DifferentialDrive()
    assert not body.strafes
    for left, right in ((0.0, 0.0), (1.0, -1.0), (0.3, 0.7), (-0.2, -0.9), (1.0, 1.0)):
        drive = Drive.from_wheels(left, right)
        assert body.wheels(drive) == pytest.approx((left, right))


def test_the_mix_is_clipped_after_it_is_mixed() -> None:
    """The pre-stage-24 arithmetic: mix, then clip, so a saturated turn still turns."""
    body = DifferentialDrive()
    assert body.wheels(Drive(forward=0.9, turn=0.5)) == pytest.approx((0.4, 1.0))
    assert body.wheels(Drive(forward=-0.9, turn=-0.5)) == pytest.approx((-0.4, -1.0))


def test_mecanum_adds_sideways_and_folds_back_to_a_pair() -> None:
    body = MecanumDrive()
    assert body.strafes
    assert body.wheels(Drive(forward=1.0)) == pytest.approx((1.0, 1.0, 1.0, 1.0))
    assert body.wheels(Drive(turn=1.0)) == pytest.approx((-1.0, 1.0, -1.0, 1.0))
    # pure strafe: the diagonal pairs run opposite ways, the vehicle slides
    assert body.wheels(Drive(strafe=1.0)) == pytest.approx((1.0, -1.0, -1.0, 1.0))
    # what a two-wheel world sees of a pure strafe: nothing
    assert MecanumDrive.as_differential(body.wheels(Drive(strafe=1.0))) == pytest.approx((0.0, 0.0))
    # and of a plain drive: the drive
    assert MecanumDrive.as_differential(body.wheels(Drive(forward=0.5))) == pytest.approx(
        (0.5, 0.5)
    )


def test_a_brain_may_speak_either_language() -> None:
    assert drive_of((0.2, 0.8)).forward == pytest.approx(0.5)
    assert drive_of((0.2, 0.8)).turn == pytest.approx(0.3)
    assert drive_of((0.2, 0.8), fire=0.9).pulls_trigger
    intent = Drive(forward=0.1, turn=0.2, strafe=0.3, fire=0.8)
    assert drive_of(intent) is intent
    assert drive_of(Drive(forward=0.1), fire=0.7).fire == 0.7


def test_bodies_are_named() -> None:
    assert build_body("differential").name == "differential"
    assert build_body("mecanum").strafes
    with pytest.raises(ValueError, match="unknown body"):
        build_body("hovercraft")
