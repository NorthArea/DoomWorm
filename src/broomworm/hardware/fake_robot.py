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
from broomworm.hardware.link import LineLink, SimLink

# The shift-register bit map the fake car answers with (the firmware's guess, so the
# host tool can be exercised end to end): bit -> (wheel, direction).
PROBE_MAP: dict[int, tuple[str, str]] = {
    128: ("FL", "f"), 64: ("FL", "b"), 32: ("FR", "f"), 16: ("FR", "b"),
    8: ("RL", "f"), 4: ("RL", "b"), 2: ("RR", "f"), 1: ("RR", "b"),
}  # fmt: skip


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
        elif cmd == "probe":
            return self._probe(command)
        else:
            raise ValueError(f"unknown command {cmd!r}")
        raw = self.calibration.raw_from_channels(channels, tick=self.tick)
        return raw.to_dict() | {"charging": int(self.sim.world.docked)}

    def _probe(self, command: dict[str, Any]) -> dict[str, Any]:
        """Bench command: energise one shift-register pattern for ``ms`` at ``duty``.

        The fake car knows its own bit map (:data:`PROBE_MAP`) and says which wheel
        turned, which is what the operator reports on the real car.
        """
        bits = int(command.get("bits", 0))
        duty = max(0.0, min(1.0, float(command.get("duty", 0.5))))
        ticks = max(1, int(command.get("ms", 300)) // 100)
        hits = [PROBE_MAP[b] for b in PROBE_MAP if bits & b]
        left = right = 0.0
        for wheel, direction in hits:
            sign = duty if direction == "f" else -duty
            if wheel in ("FL", "RL"):
                left += sign / 2.0
            else:
                right += sign / 2.0
        channels = self.sim.reset()
        for _ in range(ticks):
            channels = self.sim.step(left, right)
        raw = self.calibration.raw_from_channels(channels, tick=self.tick)
        if len(hits) == 1:
            wheel, direction = hits[0]
        elif hits:
            wheel, direction = "multi", "?"
        else:
            wheel, direction = "none", "-"
        return raw.to_dict() | {"probe": bits, "probe_wheel": wheel, "probe_dir": direction}

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


def serve_fake(robot: FakeRobot, name: str = "fake") -> LineLink:
    """A :class:`LineLink` talking to ``robot`` over a socketpair on a daemon thread."""
    import socket
    import threading

    host_sock, robot_sock = socket.socketpair()
    robot_file = robot_sock.makefile("rw", encoding="utf-8", newline="\n")

    def run() -> None:
        try:
            robot.serve(robot_file, robot_file)
        finally:
            robot_sock.close()

    threading.Thread(target=run, daemon=True).start()
    host_file = host_sock.makefile("rw", encoding="utf-8", newline="\n")
    return LineLink(host_file, host_file, robot.calibration, name=name, transport=host_sock)
