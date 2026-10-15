"""Tests for subchannel airtime allocation and feasibility constraints."""

import numpy as np
import pytest
from semantic_noma.config import NetworkConfig
from semantic_noma.resource_allocator import ResourceAllocator


def test_airtime_constraints_edge_cases():
    """Verify airtime feasibility under extreme actions (-1, +1, zeros)."""
    config = NetworkConfig(num_users=8, num_pairs=4)
    allocator = ResourceAllocator(config)

    # 1. Action = all -1 (all preference z_g = 0)
    act_min = np.ones(12, dtype=np.float32) * -1.0
    _, _, tau_min = allocator.transform_raw_actions(act_min)
    assert np.all(tau_min == 0.0)

    # 2. Action = all +1 (all z_g = 1, two pairs per subchannel -> tau = 1/2 = 0.5)
    act_max = np.ones(12, dtype=np.float32) * 1.0
    _, _, tau_max = allocator.transform_raw_actions(act_max)
    for m, pairs in config.subchannel_pair_mapping.items():
        assert pytest.approx(1.0) == float(np.sum(tau_max[pairs]))
        for g in pairs:
            assert pytest.approx(0.5) == tau_max[g]

    # 3. Asymmetric actions: pair 0 has z=1, pair 1 has z=0
    act_asym = np.zeros(12, dtype=np.float32)
    # pair 0: a_tau = 1.0 -> z = 1.0
    act_asym[2] = 1.0
    # pair 1: a_tau = -1.0 -> z = 0.0
    act_asym[5] = -1.0
    _, _, tau_asym = allocator.transform_raw_actions(act_asym)
    assert pytest.approx(1.0) == tau_asym[0]
    assert pytest.approx(0.0) == tau_asym[1]


def test_airtime_sum_constraint_random_samples():
    """Verify sum of airtimes on each subchannel is <= 1.0 for 1000 random actions."""
    config = NetworkConfig()
    allocator = ResourceAllocator(config)
    rng = np.random.default_rng(777)

    for _ in range(1000):
        raw_act = rng.uniform(-1.0, 1.0, size=12).astype(np.float32)
        _, _, tau = allocator.transform_raw_actions(raw_act)

        # Non-negativity
        assert np.all(tau >= 0.0)
        assert np.all(tau <= 1.0)

        # Subchannel airtime sum <= 1.0
        for m, pairs in config.subchannel_pair_mapping.items():
            subchannel_airtime = float(np.sum(tau[pairs]))
            assert subchannel_airtime <= 1.0 + 1e-9
