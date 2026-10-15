"""Tests for instantaneous power constraints and power decomposition."""

import numpy as np
import pytest
from semantic_noma.config import NetworkConfig
from semantic_noma.resource_allocator import ResourceAllocator


def test_power_constraints_all_action_extremes():
    """Verify power constraints hold at action bounds [-1, 1] and random samples."""
    config = NetworkConfig(num_users=8, num_pairs=4, p_max_w=1.0)
    allocator = ResourceAllocator(config)
    channel_gains = np.ones(8) * 1e-10

    # 1. Action = all -1 (minimum power)
    act_min = np.ones(12, dtype=np.float32) * -1.0
    res_min = allocator.allocate(act_min, channel_gains)
    assert np.all(res_min.eta == 0.0)
    assert np.all(res_min.pair_powers == 0.0)
    assert res_min.power_violation == 0.0

    # 2. Action = all +1 (maximum power)
    act_max = np.ones(12, dtype=np.float32) * 1.0
    res_max = allocator.allocate(act_max, channel_gains)
    assert np.all(res_max.eta == 1.0)
    # Each subchannel has max power P_max / M = 0.5 W
    assert np.all(pytest.approx(0.5) == res_max.pair_powers)
    assert res_max.power_violation == 0.0

    # 3. 500 Random continuous actions
    rng = np.random.default_rng(42)
    for _ in range(500):
        raw_act = rng.uniform(-1.0, 1.0, size=12).astype(np.float32)
        res = allocator.allocate(raw_act, channel_gains)

        # Pair powers must never exceed P_max / M
        assert np.all(res.pair_powers <= (config.p_max_w / config.num_subchannels) + 1e-9)
        assert res.power_violation == 0.0

        # Weak and strong powers must sum to pair power
        for g in range(config.num_pairs):
            u_w = res.weak_users[g]
            u_s = res.strong_users[g]
            assert pytest.approx(res.pair_powers[g], abs=1e-9) == res.user_powers[u_w] + res.user_powers[u_s]


def test_noma_power_split_bounds():
    """Verify weak user power fraction beta_g is strictly in [0.5, 0.95]."""
    config = NetworkConfig()
    allocator = ResourceAllocator(config)
    rng = np.random.default_rng(99)

    for _ in range(200):
        raw_act = rng.uniform(-1.0, 1.0, size=12).astype(np.float32)
        _, beta, _ = allocator.transform_raw_actions(raw_act)
        assert np.all(beta >= 0.5 - 1e-9)
        assert np.all(beta <= 0.95 + 1e-9)
