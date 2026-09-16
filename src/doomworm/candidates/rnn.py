"""A small recurrent network from scratch: candidate 4 of the A2 bake-off (Plan §20.4).

Same neuron dynamics as the graded worm (leaky potential, activity clipped to
[0, 1]), same ``brain_steps`` ticks per environment step, same evolution; the
only difference is the wiring: dense input -> hidden -> hidden and a linear
readout to two wheels, all weights random at birth. It answers the question
"what does a network with no biology do with the same budget?".
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from doomworm.candidates.base import Wheels

UNBOUNDED = ("odom_x", "odom_y", "odom_heading", "gyro_heading")


class RNNBrain:
    """Leaky recurrent layer over the bounded sensor channels, linear wheel readout."""

    def __init__(
        self,
        inputs: Sequence[str],
        hidden: int = 32,
        brain_steps: int = 5,
        decay: float = 0.5,
        init_scale: float = 0.3,
        forward: float = 0.5,
        seed: int = 0,
        name: str = "rnn",
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.inputs = [c for c in inputs if c not in UNBOUNDED]
        self.hidden = hidden
        self.brain_steps = brain_steps
        self.decay = decay
        self.name = name
        self.meta = dict(meta or {}) | {"candidate": "rnn"}
        rng = np.random.default_rng(seed)
        n_in = len(self.inputs)
        self.w_in = rng.normal(0.0, init_scale, size=(hidden, n_in))
        self.w_rec = rng.normal(0.0, init_scale, size=(hidden, hidden))
        self.bias = np.zeros(hidden)
        # readout scaled by 1/sqrt(hidden) so the forward bias is not swamped at birth
        self.w_out = rng.normal(0.0, init_scale / math.sqrt(hidden), size=(2, hidden))
        self.bias_out = np.array([forward, forward])
        self.potential = np.zeros(hidden)
        self.activity_vector = np.zeros(hidden)

    # --- Brain -------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the hidden state."""
        self.potential = np.zeros(self.hidden)
        self.activity_vector = np.zeros(self.hidden)

    def act(self, channels: Mapping[str, float]) -> Wheels:
        """``brain_steps`` ticks on the same input; wheels from the mean activity."""
        x = np.array([channels.get(c, 0.0) for c in self.inputs], dtype=float)
        window = np.zeros(self.hidden)
        for _ in range(self.brain_steps):
            drive = self.w_in @ x + self.w_rec @ self.activity_vector + self.bias
            self.potential = self.potential * (1.0 - self.decay) + drive
            self.activity_vector = np.clip(self.potential, 0.0, 1.0)
            window += self.activity_vector
        mean = window / self.brain_steps
        left, right = np.clip(self.w_out @ mean + self.bias_out, -1.0, 1.0)
        return float(left), float(right)

    @property
    def activity(self) -> dict[str, float]:
        """Hidden activity by unit name (debug screen)."""
        return {f"h{i}": float(v) for i, v in enumerate(self.activity_vector)}

    # --- Trainable ----------------------------------------------------------------

    @property
    def n_weights(self) -> int:
        """Input, recurrent, bias, readout and output-bias weights."""
        return self.w_in.size + self.w_rec.size + self.bias.size + self.w_out.size + 2

    def get_weights(self) -> list[float]:
        """Flat genome in a fixed order."""
        parts = (self.w_in, self.w_rec, self.bias, self.w_out, self.bias_out)
        return [float(v) for v in np.concatenate([p.ravel() for p in parts])]

    def set_weights(self, weights: Sequence[float]) -> None:
        """Load a flat genome."""
        w = np.asarray(weights, dtype=float)
        if w.size != self.n_weights:
            raise ValueError(f"expected {self.n_weights} weights, got {w.size}")
        cut = 0
        for name in ("w_in", "w_rec", "bias", "w_out", "bias_out"):
            arr = getattr(self, name)
            setattr(self, name, w[cut : cut + arr.size].reshape(arr.shape))
            cut += arr.size

    # --- persistence ---------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """JSON with kind, architecture, weights and meta."""
        data = {
            "kind": "rnn",
            "inputs": self.inputs,
            "hidden": self.hidden,
            "brain_steps": self.brain_steps,
            "decay": self.decay,
            "weights": self.get_weights(),
            "meta": self.meta,
        }
        Path(path).write_text(json.dumps(data) + "\n")

    @classmethod
    def from_file(cls, path: Path | str) -> RNNBrain:
        """Load a saved RNN brain."""
        data = json.loads(Path(path).read_text())
        if data.get("kind") != "rnn":
            raise ValueError(f"{path} is not an rnn brain")
        brain = cls(
            data["inputs"],
            data["hidden"],
            data["brain_steps"],
            data["decay"],
            name=Path(path).stem,
            meta=data.get("meta"),
        )
        brain.set_weights(data["weights"])
        return brain
