import numpy as np
import pytest

SEED = 42


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic RNG for tests (plan rule: use deterministic seeds)."""
    return np.random.default_rng(SEED)
