import numpy as np

import doomworm
from doomworm.cli import main


def test_version_is_set() -> None:
    assert doomworm.__version__ != "0.0.0"


def test_cli_help_exits_zero(capsys) -> None:  # type: ignore[no-untyped-def]
    assert main([]) == 0
    assert "doomworm" in capsys.readouterr().out


def test_rng_fixture_is_deterministic(rng: np.random.Generator) -> None:
    assert rng.integers(0, 1_000_000) == np.random.default_rng(42).integers(0, 1_000_000)
