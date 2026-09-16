"""Neural Circuit Policy candidate (Plan §20.4 #6): the published worm-inspired architecture.

Optional ``rl`` dependency group (``ncps``, torch). ``AutoNCP`` wires sensory,
inter, command and motor neurons sparsely by rule (Lechner et al. 2020), the
cell is a closed-form continuous-time (CfC) neuron. Here it is trained by the
same evolution as every other candidate: the brain exposes its torch
parameters as the flat weight vector. One CfC step per environment step.
The two motor outputs carry a trainable forward bias (``forward``, default
0.5), the analogue of the worm's tonic AVB drive: an untrained CfC outputs
about zero and the robot never moves (stage 21.10).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from doomworm.candidates.base import Wheels
from doomworm.candidates.rnn import UNBOUNDED


class NCPBrain:
    """CfC cell over an AutoNCP wiring, two motor neurons read as wheels."""

    def __init__(
        self,
        inputs: Sequence[str],
        units: int = 32,
        seed: int = 0,
        forward: float = 0.5,
        name: str = "ncp",
        meta: dict[str, Any] | None = None,
    ) -> None:
        import torch
        from ncps.torch import CfC
        from ncps.wirings import AutoNCP

        torch.set_num_threads(1)
        torch.manual_seed(seed)
        self.inputs = [c for c in inputs if c not in UNBOUNDED]
        self.units = units
        self.seed = seed
        self.name = name
        self.meta = dict(meta or {}) | {"candidate": "ncp"}
        self.wiring = AutoNCP(units, 2, seed=seed)
        self.model = CfC(len(self.inputs), self.wiring, batch_first=True)
        self.model.eval()
        self._torch = torch
        self.bias_out = np.array([forward, forward])
        self.hidden: Any = None
        self.activity_vector = np.zeros(units)

    # --- Brain -------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the cell state."""
        self.hidden = None
        self.activity_vector = np.zeros(self.units)

    def act(self, channels: Mapping[str, float]) -> Wheels:
        """One CfC step on this tick's channels."""
        torch = self._torch
        x = torch.tensor([[[channels.get(c, 0.0) for c in self.inputs]]], dtype=torch.float32)
        with torch.no_grad():
            out, self.hidden = self.model(x, self.hidden)
        self.activity_vector = self.hidden[0].numpy().astype(float)
        left, right = np.clip(out[0, 0].numpy().astype(float) + self.bias_out, -1.0, 1.0)
        return float(left), float(right)

    @property
    def activity(self) -> dict[str, float]:
        """Cell state by unit (debug screen)."""
        return {f"n{i}": float(v) for i, v in enumerate(self.activity_vector)}

    # --- Trainable ----------------------------------------------------------------

    @property
    def n_weights(self) -> int:
        """Every torch parameter of the cell plus the two output biases."""
        return int(sum(p.numel() for p in self.model.parameters())) + 2

    def get_weights(self) -> list[float]:
        """Flat parameter vector (parameters in module order)."""
        torch = self._torch
        flat = torch.cat([p.detach().reshape(-1) for p in self.model.parameters()])
        return [float(v) for v in flat] + [float(v) for v in self.bias_out]

    def set_weights(self, weights: Sequence[float]) -> None:
        """Load a flat parameter vector."""
        w = np.asarray(weights, dtype=float)
        if w.size != self.n_weights:
            raise ValueError(f"expected {self.n_weights} weights, got {w.size}")
        cut = 0
        with self._torch.no_grad():
            for p in self.model.parameters():
                n = p.numel()
                chunk = self._torch.tensor(w[cut : cut + n], dtype=p.dtype).reshape(p.shape)
                p.copy_(chunk)
                cut += n
        self.bias_out = w[cut : cut + 2].copy()

    # --- persistence ---------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """JSON with kind, wiring seed, inputs, weights and meta."""
        data = {
            "kind": "ncp",
            "inputs": self.inputs,
            "units": self.units,
            "seed": self.seed,
            "weights": self.get_weights(),
            "meta": self.meta,
        }
        Path(path).write_text(json.dumps(data) + "\n")

    @classmethod
    def from_file(cls, path: Path | str) -> NCPBrain:
        """Rebuild the wiring from its seed and load the weights."""
        data = json.loads(Path(path).read_text())
        if data.get("kind") != "ncp":
            raise ValueError(f"{path} is not an ncp brain")
        brain = cls(
            data["inputs"], data["units"], data["seed"], name=Path(path).stem, meta=data.get("meta")
        )
        brain.set_weights(data["weights"])
        return brain
