"""Layers that sit outside the brain and give it a sense it does not have.

A layer may add channels; it never chooses an action. Wrapping a brain in one
is a condition of the experiment, reported as its own row.
"""

from doomworm.layer.memory import NOVELTY, MemoryLayer

__all__ = ["NOVELTY", "MemoryLayer"]
