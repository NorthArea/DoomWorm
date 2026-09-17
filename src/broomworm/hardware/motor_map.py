"""Bench tool for the QA052 shield: which shift-register bit turns which wheel (stage 23.2).

The shield sets every motor's direction from one byte written to a shift
register; the ACEBOTT sample only documents one motor (128 forward, 64
backward). ``broomworm motor-map`` energises the eight bits one at a time
through the ``probe`` command of the protocol, asks which wheel turned and in
which direction, and prints the ``MOTOR_FWD`` / ``MOTOR_BWD`` tables to paste
into ``firmware/esp32_car/esp32_car.ino``. A fake car answers by itself, which
is how the tool is tested.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

WHEELS = ("FL", "FR", "RL", "RR")  # firmware motor indices 0..3
BITS = (128, 64, 32, 16, 8, 4, 2, 1)
Answer = tuple[str, str]  # (wheel or "none"/"multi", "f"/"b"/"-")


@dataclass
class MotorMapReport:
    """What every bit did."""

    observed: dict[int, Answer] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    def tables(self) -> tuple[list[int], list[int]]:
        """``MOTOR_FWD`` and ``MOTOR_BWD`` by wheel index; 0 where nothing was found."""
        fwd = [0] * 4
        bwd = [0] * 4
        for bit, (wheel, direction) in self.observed.items():
            if wheel not in WHEELS:
                continue
            i = WHEELS.index(wheel)
            if direction == "f":
                fwd[i] |= bit
            elif direction == "b":
                bwd[i] |= bit
        return fwd, bwd

    def check(self) -> None:
        """Fill :attr:`problems`."""
        self.problems = []
        for bit, (wheel, _direction) in self.observed.items():
            if wheel == "multi":
                self.problems.append(f"bit {bit} moved more than one wheel")
        fwd, bwd = self.tables()
        for i, wheel in enumerate(WHEELS):
            if fwd[i] == 0:
                self.problems.append(f"no bit drives {wheel} forward")
            if bwd[i] == 0:
                self.problems.append(f"no bit drives {wheel} backward")
            if fwd[i] and bwd[i] and fwd[i] == bwd[i]:
                self.problems.append(f"{wheel}: the same bit reported for both directions")

    def c_tables(self) -> str:
        """The two lines for the sketch."""
        fwd, bwd = self.tables()
        return (
            f"static const uint8_t MOTOR_FWD[4] = {{{fwd[0]}, {fwd[1]}, {fwd[2]}, {fwd[3]}}};\n"
            f"static const uint8_t MOTOR_BWD[4] = {{{bwd[0]}, {bwd[1]}, {bwd[2]}, {bwd[3]}}};\n"
        )

    def markdown(self) -> str:
        """Report."""
        lines = ["| bit | wheel | direction |", "|---|---|---|"]
        for bit in BITS:
            wheel, direction = self.observed.get(bit, ("?", "?"))
            lines.append(f"| {bit} | {wheel} | {direction} |")
        lines += ["", "```c", self.c_tables().rstrip("\n"), "```", ""]
        lines += [f"PROBLEM: {p}" for p in self.problems] or [
            "OK: every wheel has one bit per direction"
        ]
        return "\n".join(lines) + "\n"


def answer_from_reply(reply: dict[str, Any]) -> Answer | None:
    """A fake car reports the wheel itself; a real one leaves it to the operator."""
    if "probe_wheel" in reply:
        return str(reply["probe_wheel"]), str(reply.get("probe_dir", "-"))
    return None


def ask_operator(bit: int, _reply: dict[str, Any]) -> Answer:
    """Interactive answer: ``FL f`` / ``RR b`` / ``none``."""
    while True:
        text = input(f"bit {bit}: which wheel turned, which way? (FL|FR|RL|RR f|b, none, multi) ")
        parts = text.strip().split()
        if not parts:
            continue
        wheel = parts[0].upper()
        if wheel in ("NONE", "MULTI"):
            return wheel.lower(), "-" if wheel == "NONE" else "?"
        if wheel in WHEELS and len(parts) == 2 and parts[1].lower() in ("f", "b"):
            return wheel, parts[1].lower()
        print("  answer like: FL f   or   RR b   or   none")


def motor_map(
    link: Any,
    duty: float = 0.5,
    ms: int = 400,
    answer: Callable[[int, dict[str, Any]], Answer] = ask_operator,
    bits: tuple[int, ...] = BITS,
) -> MotorMapReport:
    """Probe every bit; answers come from the reply (fake car) or from ``answer``."""
    report = MotorMapReport()
    for bit in bits:
        reply = link.command({"cmd": "probe", "bits": bit, "duty": duty, "ms": ms})
        found = answer_from_reply(reply)
        report.observed[bit] = found if found is not None else answer(bit, reply)
    report.check()
    return report
