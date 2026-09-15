"""The robot link: the same loop drives the simulator and the machine (Plan §20.5).

A :class:`RobotLink` is what the closed loop sees: ``reset()`` returns the
first channel frame, ``step(left, right)`` sends wheel commands for one tick
and returns the next frame. :class:`SimLink` is the 2D world behind that
interface; :class:`LineLink` is a real machine behind a JSON-lines text
stream (TCP over Wi-Fi to an ESP32, a serial port wrapped as a text file,
or a socketpair to :class:`~doomworm.hardware.fake_robot.FakeRobot`).

Wire protocol, one JSON object per line, host -> robot then robot -> host:

    {"cmd": "reset"}                              -> reading
    {"cmd": "drive", "left": l, "right": r}       -> reading   (l, r in [-1, 1])

A reading is :class:`~doomworm.hardware.calibration.RawReading` as JSON, in
physical units; the link converts it with a :class:`Calibration`.
"""

from __future__ import annotations

import contextlib
import json
import socket
from typing import Any, Protocol, TextIO, runtime_checkable

from doomworm.environments.sensors import PRESETS, SensorConfig, SensorSuite
from doomworm.environments.simple_2d import World
from doomworm.hardware.calibration import Calibration, RawReading


@runtime_checkable
class RobotLink(Protocol):
    """Channels out, wheels in, one tick at a time."""

    name: str

    @property
    def channel_names(self) -> list[str]:
        """Every channel a frame carries, in a fixed order."""
        ...

    def reset(self) -> dict[str, float]:
        """Start an episode; return the frame at rest."""
        ...

    def step(self, left: float, right: float) -> dict[str, float]:
        """Apply wheels for one tick; return the next frame."""
        ...

    def truth(self) -> tuple[float, float, float] | None:
        """Ground-truth pose (x, y, heading) when known (simulator), else ``None``."""
        ...

    def close(self) -> None:
        """Release the transport."""
        ...

    def meta(self) -> dict[str, Any]:
        """What to record in a drive log header."""
        ...


def _clip(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


class SimLink:
    """The 2D world and its sensor suite behind the link interface."""

    name = "sim"

    def __init__(
        self,
        world: World,
        sensors: SensorConfig | str = "vacuum",
        sensor_seed: int = 0,
        seed: int | None = None,
        maps: str = "apartment",
        task: str = "clean",
    ) -> None:
        cfg = PRESETS[sensors] if isinstance(sensors, str) else sensors
        self.world = world
        self.suite = SensorSuite(cfg, seed=sensor_seed)
        self.seed, self.maps, self.task = seed, maps, task
        self._obs = world.observe()

    @property
    def channel_names(self) -> list[str]:
        """Suite channels."""
        return self.suite.channel_names

    def reset(self) -> dict[str, float]:
        """Same warm start as :func:`doomworm.episode.run_brain_episode`."""
        self._obs = self.world.observe()
        self.suite.reset(self.world)
        return self.suite.read(self.world, self._obs)

    def step(self, left: float, right: float) -> dict[str, float]:
        """One world tick."""
        self._obs = self.world.step(_clip(left), _clip(right))
        return self.suite.read(self.world, self._obs)

    @property
    def done(self) -> bool:
        """Episode over (the world's death condition)."""
        return self.world.dead

    def truth(self) -> tuple[float, float, float]:
        """Exact pose."""
        a = self.world.agent
        return (a.x, a.y, a.heading)

    def close(self) -> None:
        """Nothing to release."""

    def meta(self) -> dict[str, Any]:
        """Log header fields."""
        return {
            "link": self.name,
            "sensors": self.suite.config.name,
            "sensor_seed": self.suite.seed,
            "seed": self.seed,
            "maps": self.maps,
            "task": self.task,
        }


class LineLink:
    """A machine (or a fake one) behind a JSON-lines text stream."""

    def __init__(
        self,
        reader: TextIO,
        writer: TextIO,
        calibration: Calibration | None = None,
        name: str = "line",
        transport: socket.socket | None = None,
    ) -> None:
        self.reader, self.writer = reader, writer
        self.transport = transport
        self.calibration = calibration or Calibration()
        self.name = name
        self.raw_last: RawReading | None = None
        self._names = SensorSuite(self.calibration.sensors).channel_names

    @property
    def channel_names(self) -> list[str]:
        """Channels of the calibrated preset."""
        return self._names

    def _exchange(self, command: dict[str, Any]) -> dict[str, float]:
        self.writer.write(json.dumps(command) + "\n")
        self.writer.flush()
        line = self.reader.readline()
        if not line:
            raise ConnectionError(f"{self.name}: the robot closed the link")
        data = json.loads(line)
        if not isinstance(data, dict) or "ranges_m" not in data:
            raise ValueError(f"{self.name}: expected a reading with ranges_m, got {line.strip()!r}")
        self.raw_last = RawReading.from_dict(data)
        return self.calibration.channels(self.raw_last)

    def reset(self) -> dict[str, float]:
        """Ask the robot to zero its odometry and send the frame at rest."""
        return self._exchange({"cmd": "reset"})

    def step(self, left: float, right: float) -> dict[str, float]:
        """Send wheels for one tick, get the next frame."""
        return self._exchange({"cmd": "drive", "left": _clip(left), "right": _clip(right)})

    def truth(self) -> None:
        """A real machine has no ground truth."""
        return None

    def close(self) -> None:
        """Tell the robot to stop, then close the stream."""
        try:
            self.writer.write(json.dumps({"cmd": "drive", "left": 0.0, "right": 0.0}) + "\n")
            self.writer.flush()
        except (OSError, ValueError):
            pass
        for stream in {self.reader, self.writer}:
            with contextlib.suppress(OSError, ValueError):
                stream.close()
        if self.transport is not None:
            with contextlib.suppress(OSError):
                self.transport.close()

    def meta(self) -> dict[str, Any]:
        """Log header fields."""
        return {"link": self.name, "sensors": self.calibration.sensors.name} | {
            "calibration": self.calibration.to_dict()
        }


def connect_tcp(host: str, port: int, calibration: Calibration | None = None) -> LineLink:
    """Open the JSON-lines protocol over TCP (an ESP32 / Raspberry Pi serving on ``port``)."""
    sock = socket.create_connection((host, port), timeout=10.0)
    stream = sock.makefile("rw", encoding="utf-8", newline="\n")
    return LineLink(stream, stream, calibration, name=f"tcp://{host}:{port}", transport=sock)
