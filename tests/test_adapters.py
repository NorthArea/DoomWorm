"""Sensory and motor adapters (Plan §2.2, §43)."""

from doomworm.adapters import MotorAdapter, SensoryAdapter


def test_sensor_mapping() -> None:
    adapter = SensoryAdapter(
        channels={"sensor_front": ("S_FRONT", 2.0), "sensor_left": ("S_LEFT", 1.0)},
        tonic={"M_LEFT": 1.0},
    )
    currents = adapter({"sensor_front": 0.25, "sensor_left": 0.0, "ignored": 9.0})
    assert currents == {"S_FRONT": 0.5, "S_LEFT": 0.0, "M_LEFT": 1.0}


def test_sensor_mapping_missing_channel_is_zero() -> None:
    adapter = SensoryAdapter(channels={"sensor_front": ("S_FRONT", 1.0)})
    assert adapter({}) == {"S_FRONT": 0.0}


def test_motor_mapping() -> None:
    adapter = MotorAdapter(left="M_LEFT", right="M_RIGHT")
    assert adapter({"M_LEFT": 1.0, "M_RIGHT": 0.0, "S_FRONT": 1.0}) == (1.0, 0.0)
    assert adapter({}) == (0.0, 0.0)
