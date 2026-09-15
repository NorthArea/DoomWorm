"""Phase A3 (Plan §20.5): the machine behind the same loop as the simulator.

Only the Environment and the adapters change between the 2D world and the
robot; the brain is the same file. This package holds the link interface,
the units contract, the drive/teleop/record loop and the sim-vs-real check.
"""

from doomworm.hardware.calibration import Calibration, RawReading
from doomworm.hardware.compare import ComparisonReport, compare_logs, replay_in_sim
from doomworm.hardware.drive import DriveRow, Teleop, drive, raw_keys, read_drive_log
from doomworm.hardware.fake_robot import FakeRobot
from doomworm.hardware.link import LineLink, RobotLink, SimLink, connect_tcp

__all__ = [
    "Calibration",
    "ComparisonReport",
    "DriveRow",
    "FakeRobot",
    "LineLink",
    "RawReading",
    "RobotLink",
    "SimLink",
    "Teleop",
    "compare_logs",
    "connect_tcp",
    "drive",
    "raw_keys",
    "read_drive_log",
    "replay_in_sim",
]
