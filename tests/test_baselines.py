"""Tests for baseline policies feasibility and evaluation execution."""

import numpy as np
import pytest
from semantic_noma.baselines import (
    EqualAllocationPolicy,
    RandomAllocationPolicy,
    SemanticPriorityPolicy,
)
from semantic_noma.config import ProjectConfig
from semantic_noma.environment import SemanticNomaEnv
from semantic_noma.evaluation import run_episode


def test_equal_allocation_feasibility():
    """Verify Equal Allocation produces valid action and exact equal parameters."""
    config = ProjectConfig()
    policy = EqualAllocationPolicy(config=config)
    env = SemanticNomaEnv(config=config)
    obs, _ = env.reset(seed=42)

    action, _ = policy.predict(obs)
    assert action.shape == (12,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)

    # Check allocation directly
    res = env.allocator.allocate(action, env.current_channels)
    assert np.all(pytest.approx(0.5) == res.airtimes)
    assert np.all(pytest.approx(1.0) == res.eta)
    assert np.all(pytest.approx(0.75, abs=1e-4) == res.beta)
    assert res.power_violation == 0.0
    assert res.airtime_violation == 0.0


def test_semantic_priority_feasibility():
    """Verify Semantic Priority heuristic produces feasible allocations and respects power/airtime limits."""
    config = ProjectConfig()
    policy = SemanticPriorityPolicy(config=config)
    env = SemanticNomaEnv(config=config)
    obs, _ = env.reset(seed=42)

    action, _ = policy.predict(obs)
    assert action.shape == (12,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)

    res = env.allocator.allocate(action, env.current_channels)
    assert res.power_violation == 0.0
    assert res.airtime_violation == 0.0


def test_baselines_run_episode():
    """Verify all three baselines complete a full evaluation episode and return valid metrics."""
    config = ProjectConfig()
    env = SemanticNomaEnv(config=config)

    baselines = [
        EqualAllocationPolicy(config),
        SemanticPriorityPolicy(config),
        RandomAllocationPolicy(seed=42, config=config),
    ]

    for b in baselines:
        metrics, _ = run_episode(env=env, policy=b, seed=2000, policy_name=b.__class__.__name__)
        assert np.isfinite(metrics.episode_reward)
        assert np.isfinite(metrics.semantic_utility)
        assert metrics.max_power_violation == 0.0
        assert metrics.max_airtime_violation == 0.0
        assert 0.0 <= metrics.task_completion_rate <= 1.0
