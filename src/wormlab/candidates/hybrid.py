"""Hybrids: a frozen reflex brain under a trainable policy (Plan §20.4 item 7, axis C7).

The question of the axis is whether the connectome is worth anything as a
*part* of a learner rather than as the whole of it. The hybrid answers it in
the cheapest form the platform allows: a brain that is already trained (the
curriculum worm) runs every step with its weights frozen, its wheel commands
(and trigger) are handed to a small network from scratch as three extra
channels, and only that network is trained. The genome is the policy's, so the
training harness, the benchmark and the layer stay untouched.

    doomworm evolve --candidate hybrid --init-brain docs/brains/a2/worm.json

With the reflex removed the candidate is exactly the ``rnn`` row, which is what
makes the comparison in `docs/findings.md` a fair one.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from wormlab.candidates.base import Trainable, Wheels
from wormlab.candidates.rnn import RNNBrain

if TYPE_CHECKING:
    from wormlab.candidates.registry import TrainableBrain

REFLEX_CHANNELS = ("reflex_left", "reflex_right", "reflex_fire")


class HybridBrain:
    """A frozen reflex brain plus a trainable policy that sees its output."""

    def __init__(
        self,
        reflex: TrainableBrain,
        policy: RNNBrain,
        reflex_path: str,
        name: str = "hybrid",
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.reflex = reflex
        self.policy = policy
        self.reflex_path = reflex_path
        self.name = name
        self.meta = dict(meta or {}) | {"candidate": "hybrid", "reflex": reflex_path}
        self.fire = 0.0
        self.last_reflex: Wheels = (0.0, 0.0)
        self.last_channels: dict[str, float] = {}

    # --- Brain -------------------------------------------------------------------

    def reset(self) -> None:
        """Clear both halves."""
        self.reflex.reset()
        self.policy.reset()
        self.fire = 0.0
        self.last_reflex = (0.0, 0.0)
        self.last_channels = {}

    def act(self, channels: Mapping[str, float]) -> Wheels:
        """Reflex first, then the policy on the sensors plus the reflex's answer."""
        left, right = self.reflex.act(channels)
        self.last_reflex = (left, right)
        self.last_channels = dict(channels) | {
            "reflex_left": left,
            "reflex_right": right,
            "reflex_fire": float(getattr(self.reflex, "fire", 0.0)),
        }
        wheels = self.policy.act(self.last_channels)
        self.fire = (
            self.policy.fire if self.policy.outputs == 3 else self.last_channels["reflex_fire"]
        )
        return wheels

    @property
    def activity(self) -> dict[str, float]:
        """Policy activity plus the reflex's wheels (debug screen)."""
        return dict(self.policy.activity) | {
            "reflex_left": self.last_reflex[0],
            "reflex_right": self.last_reflex[1],
        }

    # --- Trainable ----------------------------------------------------------------

    @property
    def n_weights(self) -> int:
        """Only the policy is trained; the reflex's synapses are frozen."""
        return self.policy.n_weights

    def get_weights(self) -> list[float]:
        """The policy's genome."""
        return self.policy.get_weights()

    def set_weights(self, weights: Sequence[float]) -> None:
        """Load a policy genome."""
        self.policy.set_weights(weights)

    # --- persistence ---------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """JSON with the reflex's path (it is never modified) and the policy inline."""
        data = {
            "kind": "hybrid",
            "reflex": self.reflex_path,
            "policy": self.policy.to_dict(),
            "meta": self.meta,
        }
        Path(path).write_text(json.dumps(data) + "\n")

    @classmethod
    def from_file(cls, path: Path | str, **overrides: Any) -> HybridBrain:
        """Load a hybrid; ``overrides`` reach the reflex's scenario (maps, task, sensors)."""
        from wormlab.candidates.registry import load_candidate

        data = json.loads(Path(path).read_text())
        if data.get("kind") != "hybrid":
            raise ValueError(f"{path} is not a hybrid brain")
        meta = dict(data.get("meta", {}))
        params = dict(meta.get("reflex_params", {})) | overrides
        reflex = load_candidate(data["reflex"], **params)
        return cls(
            cast("TrainableBrain", reflex),
            RNNBrain.from_dict(data["policy"]),
            data["reflex"],
            name=Path(path).stem,
            meta=meta,
        )


def build_hybrid(
    reflex_path: str,
    inputs: Sequence[str],
    outputs: int = 2,
    seed: int = 0,
    world: Mapping[str, Any] | None = None,
    **policy_params: Any,
) -> HybridBrain:
    """A hybrid over a saved brain: its wheels become three channels of a fresh policy."""
    from wormlab.candidates.registry import load_candidate

    params = dict(world or {})
    reflex = load_candidate(reflex_path, **params)
    if not isinstance(reflex, Trainable):
        raise ValueError(f"{reflex_path} is not a candidate brain")
    policy = RNNBrain(
        [*inputs, *REFLEX_CHANNELS], seed=seed, outputs=outputs, name="hybrid", **policy_params
    )
    return HybridBrain(
        cast("TrainableBrain", reflex),
        policy,
        str(reflex_path),
        meta={"reflex_params": params, "policy": "rnn"},
    )
