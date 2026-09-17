"""A machine that is really the simulator: the reference for the firmware (Plan §20.5).

:class:`FakeRobot` answers the JSON-lines protocol of
:class:`~broomworm.hardware.link.LineLink` from a :class:`SimLink`, converting
the suite's channels back into physical units. It is what the ESP32 / Raspberry
Pi firmware has to imitate, and what the tests use as the far end of the wire.
"""

from __future__ import annotations

import contextlib
import json
from typing import Any, TextIO

from broomworm.hardware.calibration import Calibration
from broomworm.hardware.link import SimLink


class FakeRobot:
    """Serve one link session from the simulator."""

    def __init__(self, sim: SimLink, calibration: Calibration | None = None) -> None:
        self.sim = sim
        self.calibration = calibration or Calibration(sensors=sim.suite.config)
        self.tick = 0

    def handle(self, command: dict[str, Any]) -> dict[str, Any]:
        """One command -> one reading (physical units)."""
        cmd = command.get("cmd")
        if cmd == "reset":
            self.tick = 0
            channels = self.sim.reset()
        elif cmd == "drive":
            self.tick += 1
            channels = self.sim.step(
                float(command.get("left", 0.0)), float(command.get("right", 0.0))
            )
        else:
            raise ValueError(f"unknown command {cmd!r}")
        raw = self.calibration.raw_from_channels(channels, tick=self.tick)
        return raw.to_dict() | {"charging": int(self.sim.world.docked)}

    def serve(self, reader: TextIO, writer: TextIO) -> None:
        """Loop until the host closes the stream."""
        try:
            for line in reader:
                if not line.strip():
                    continue
                reply = self.handle(json.loads(line))
                writer.write(json.dumps(reply) + "\n")
                writer.flush()
        except (OSError, ValueError):
            pass
        finally:
            for stream in {reader, writer}:
                with contextlib.suppress(OSError, ValueError):
                    stream.close()
