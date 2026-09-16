"""Phase A3 (Plan §20.5): the machine behind the same loop as the simulator.

Only the Environment and the adapters change between the 2D world and the
robot; the brain is the same file. This package holds the link interface,
the units contract, the drive/teleop/record loop and the sim-vs-real check.
"""

from doomworm.hardware.calibrate import calibrate, load_calibration, save_calibration
from doomworm.hardware.calibration import Calibration, RawReading
from doomworm.hardware.compare import ComparisonReport, compare_logs, replay_in_sim
from doomworm.hardware.drive import DriveRow, Teleop, drive, raw_keys, read_drive_log
from doomworm.hardware.fake_robot import FakeRobot
from doomworm.hardware.link import LineLink, RobotLink, SimLink, connect_tcp
from doomworm.hardware.plot import plot_drive_log
from doomworm.hardware.room import Room, load_room, room_world
from doomworm.hardware.selftest import SelfTestReport, selftest

__all__ = [
    "Calibration",
    "ComparisonReport",
    "DriveRow",
    "FakeRobot",
    "LineLink",
    "RawReading",
    "RobotLink",
    "Room",
    "SelfTestReport",
    "SimLink",
    "Teleop",
    "calibrate",
    "compare_logs",
    "connect_tcp",
    "drive",
    "load_calibration",
    "load_room",
    "plot_drive_log",
    "raw_keys",
    "read_drive_log",
    "replay_in_sim",
    "room_world",
    "save_calibration",
    "selftest",
]
