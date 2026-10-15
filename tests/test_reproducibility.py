"""Tests for trajectory reproducibility given scenario seeds."""

import numpy as np
import pytest
from semantic_noma.config import ProjectConfig
from semantic_noma.environment import SemanticNomaEnv


def test_seed_reproducibility():
    """Verify that identical seeds produce identical observations, rewards, and task sequences."""
    config = ProjectConfig()
    env1 = SemanticNomaEnv(config=config)
    env2 = SemanticNomaEnv(config=config)

    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    np.testing.assert_allclose(obs1, obs2, rtol=1e-6)

    # Use a fixed sequence of actions
    rng = np.random.default_rng(999)
    actions = [rng.uniform(-1.0, 1.0, size=12).astype(np.float32) for _ in range(15)]

    for a in actions:
        o1, r1, t1, tr1, i1 = env1.step(a)
        o2, r2, t2, tr2, i2 = env2.step(a)

        np.testing.assert_allclose(o1, o2, rtol=1e-6)
        assert pytest.approx(r1, abs=1e-6) == r2
        assert t1 == t2
        assert tr1 == tr2
        assert i1["semantic_utility"] == i2["semantic_utility"]
        assert i1["useful_bits_sum"] == i2["useful_bits_sum"]
