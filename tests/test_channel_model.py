"""Tests for channel model, path loss equations, and Rayleigh fading."""

import math
import numpy as np
import pytest
from semantic_noma.channel_model import (
    ChannelModel,
    compute_free_space_path_loss,
    compute_path_loss,
)
from semantic_noma.config import NetworkConfig


def test_free_space_path_loss():
    """Verify PL_0 calculation at 1m reference distance."""
    fc = 3.5e9
    d0 = 1.0
    c = 3.0e8
    # Hand calculation: 4 * pi * 1.0 * 3.5e9 / 3.0e8 = 4 * pi * 35 / 3 = 146.607657
    # 20 * log10(146.607657) = 43.3229 dB
    pl_0 = compute_free_space_path_loss(fc, d0, c)
    expected = 20.0 * math.log10(4.0 * math.pi * 3.5e9 / 3.0e8)
    assert pytest.approx(expected, rel=1e-5) == pl_0
    assert 43.0 < pl_0 < 44.0


def test_path_loss_distance_monotonicity():
    """Verify that path loss strictly increases with distance."""
    pl_0 = 43.32
    n = 3.5
    pl_30 = compute_path_loss(30.0, pl_0, n, 1.0)
    pl_100 = compute_path_loss(100.0, pl_0, n, 1.0)
    pl_250 = compute_path_loss(250.0, pl_0, n, 1.0)

    assert pl_30 < pl_100 < pl_250

    with pytest.raises(ValueError):
        compute_path_loss(-10.0, pl_0, n, 1.0)


def test_channel_model_generation():
    """Verify ChannelModel outputs correct shape, non-negative finite gains."""
    config = NetworkConfig(num_users=8)
    model = ChannelModel(config, seed=42)
    gains = model.step()

    assert gains.shape == (8,)
    assert np.all(np.isfinite(gains))
    assert np.all(gains > 0.0)


def test_rayleigh_fading_mean_power():
    """Verify that Rayleigh fading power |h|^2 has expected value close to 1.0."""
    config = NetworkConfig(num_users=8)
    model = ChannelModel(config, seed=12345)

    samples = [model.sample_fading_power() for _ in range(5000)]
    stacked = np.concatenate(samples)
    mean_val = float(np.mean(stacked))

    # Expectation of Exp(1) is 1.0; 40,000 samples should be within 2%
    assert pytest.approx(1.0, abs=0.03) == mean_val
