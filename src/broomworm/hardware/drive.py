"""Drive loop, teleop and the drive log (Plan §20.5: manual control, sensor recording).

The log is JSON lines: a header ``{"meta": {...}}`` then one ``DriveRow`` per
tick with the wheels sent, the channels the brain saw *before* sending them,
the raw physical reading when the link had one, and the true pose when the
link knew it (simulator only). A log replays in the simulator with
:func:`broomworm.hardware.compare.replay_in_sim`.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from broomworm.body import DifferentialDrive, drive_of
from broomworm.episode import BrainLike
from broomworm.hardware.link import RobotLink

Wheels = tuple[float, float]

KEYS: dict[str, Wheels] = {
    "w": (1.0, 1.0),
    "s": (-1.0, -1.0),
    "a": (-0.6, 0.6),
    "d": (0.6, -0.6),
    "x": (0.0, 0.0),
    " ": (0.0, 0.0),
}


@dataclass
class DriveRow:
    """One tick of a drive."""

    tick: int
    wheels: Wheels
    channels: dict[str, float]
    raw: dict[str, Any] | None = None
    truth: tuple[float, float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON line."""
        return {
            "tick": self.tick,
            "wheels": list(self.wheels),
            "channels": self.channels,
            "raw": self.raw,
            "truth": None if self.truth is None else list(self.truth),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DriveRow:
        """Parse a JSON line."""
        truth = data.get("truth")
        return cls(
            tick=int(data["tick"]),
            wheels=(float(data["wheels"][0]), float(data["wheels"][1])),
            channels={k: float(v) for k, v in data["channels"].items()},
            raw=data.get("raw"),
            truth=None if truth is None else (float(truth[0]), float(truth[1]), float(truth[2])),
        )


class Teleop:
    """Manual control from a text stream: ``w a s d x`` keys or ``left right`` pairs.

    A blank line repeats the last command, ``q`` (or end of input) stops the
    drive. Works with a piped file, a script or an interactive terminal line by
    line; :func:`raw_keys` turns a tty into such a stream without Enter.
    """

    name = "teleop"

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream if stream is not None else sys.stdin
        self.last: Wheels = (0.0, 0.0)
        self.stopped = False

    def reset(self) -> None:
        """Start stopped."""
        self.last = (0.0, 0.0)
        self.stopped = False

    def act(self, channels: Mapping[str, float]) -> Wheels:
        """Next command from the stream; (0, 0) once stopped."""
        if self.stopped:
            return 0.0, 0.0
        line = self.stream.readline()
        if not line:
            self.stopped = True
            return 0.0, 0.0
        text = line.rstrip("\n")
        if text.strip() == "":
            return self.last
        if text.strip() == "q":
            self.stopped = True
            return 0.0, 0.0
        if text.strip() in KEYS:
            self.last = KEYS[text.strip()]
        else:
            left, right = (float(v) for v in text.split()[:2])
            self.last = (max(-1.0, min(1.0, left)), max(-1.0, min(1.0, right)))
        return self.last


def drive(
    link: RobotLink,
    controller: BrainLike,
    steps: int,
    log: Path | str | None = None,
    meta: Mapping[str, Any] | None = None,
    on_tick: Callable[[DriveRow], None] | None = None,
) -> list[DriveRow]:
    """Closed loop over a link: frame -> controller -> wheels -> link, recorded per tick.

    Stops after ``steps`` ticks, when a :class:`Teleop` runs out of commands, or
    when a :class:`~broomworm.hardware.link.SimLink` reports the episode over.
    """
    rows: list[DriveRow] = []
    controller.reset()
    channels = link.reset()
    header = dict(meta or {}) | link.meta() | {"controller": getattr(controller, "name", "?")}
    out: TextIO | None = None
    if log is not None:
        path = Path(log)
        path.parent.mkdir(parents=True, exist_ok=True)
        out = path.open("w", encoding="utf-8")
        out.write(json.dumps({"meta": header}) + "\n")
    try:
        for tick in range(steps):
            intent = drive_of(controller.act(channels), float(getattr(controller, "fire", 0.0)))
            if getattr(controller, "stopped", False):
                break
            # The wire still carries a wheel pair; a four-wheel machine mixes the same
            # intent on its own side (stage 24, `broomworm.body`).
            left, right = DifferentialDrive().wheels(intent)
            raw = getattr(link, "raw_last", None)
            row = DriveRow(
                tick=tick,
                wheels=(left, right),
                channels=dict(channels),
                raw=None if raw is None else raw.to_dict(),
                truth=link.truth(),
            )
            rows.append(row)
            if out is not None:
                out.write(json.dumps(row.to_dict()) + "\n")
            if on_tick is not None:
                on_tick(row)
            channels = link.step(left, right)
            if getattr(link, "done", False):
                break
    finally:
        if out is not None:
            out.close()
    return rows


def read_drive_log(path: Path | str) -> tuple[dict[str, Any], list[DriveRow]]:
    """Header meta and rows of a drive log."""
    meta: dict[str, Any] = {}
    rows: list[DriveRow] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            if "meta" in data and "tick" not in data:
                meta = dict(data["meta"])
            else:
                rows.append(DriveRow.from_dict(data))
    return meta, rows


def raw_keys(tty: TextIO | None = None) -> TextIO:
    """Turn an interactive terminal into a line-per-key stream for :class:`Teleop`.

    Puts the tty in cbreak mode for the life of the returned stream (restored on
    close). Not used by tests; ``broomworm drive --teleop`` picks it when stdin is a tty.
    """
    import termios
    import tty as ttymod

    source = tty if tty is not None else sys.stdin
    fd = source.fileno()
    saved = termios.tcgetattr(fd)
    ttymod.setcbreak(fd)

    class _Keys:
        def readline(self) -> str:
            ch = source.read(1)
            return "" if ch == "" else ch + "\n"

        def close(self) -> None:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)

    return _Keys()  # type: ignore[return-value]
