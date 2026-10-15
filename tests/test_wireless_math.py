"""Tests for noise power, NOMA SINR, achievable rate, and energy calculations."""

import math
import numpy as np
import pytest
from semantic_noma.wireless_math import (
    compute_achievable_rates,
    compute_frame_energy,
    compute_noma_sinr,
    compute_noise_power,
    decompose_pair_power,
)


def test_noise_power_calculation():
    """Validate noise power against hand-calculated reference value.
    
    N_0 = -174 dBm/Hz = 10^(-20.4) W/Hz = 3.9810717e-21 W/Hz
    B = 180,000 Hz
    NF = 7 dB -> 10^0.7 = 5.0118723
    N_m = 3.9810717e-21 * 180,000 * 5.0118723 ≈ 3.59146e-15 W
    """
    n_m = compute_noise_power(
        thermal_noise_density_dbm_hz=-174.0,
        bandwidth_hz=180000.0,
        noise_figure_db=7.0,
    )
    expected_hand_calc = 10.0 ** (-20.4) * 180000.0 * (10.0 ** 0.7)
    assert pytest.approx(expected_hand_calc, rel=1e-5) == n_m
    assert pytest.approx(3.59146e-15, rel=1e-4) == n_m


def test_decompose_pair_power():
    """Verify pair power decomposes into weak and strong users and sums to P_g."""
    p_max = 1.0
    num_subchannels = 2  # Max pair power = 0.5 W

    # Case 1: eta = 0.8, beta = 0.7
    p_g, p_w, p_s = decompose_pair_power(eta_g=0.8, beta_g=0.7, p_max_w=p_max, num_subchannels=num_subchannels)
    assert pytest.approx(0.4) == p_g
    assert pytest.approx(0.28) == p_w
    assert pytest.approx(0.12) == p_s
    assert pytest.approx(p_g) == p_w + p_s

    # Case 2: Out-of-bound beta clipped to [0.5, 0.95]
    _, p_w_low, p_s_low = decompose_pair_power(eta_g=1.0, beta_g=0.2, p_max_w=p_max, num_subchannels=num_subchannels)
    assert pytest.approx(0.25) == p_w_low
    assert pytest.approx(0.25) == p_s_low

    _, p_w_high, p_s_high = decompose_pair_power(eta_g=1.0, beta_g=0.99, p_max_w=p_max, num_subchannels=num_subchannels)
    assert pytest.approx(0.5 * 0.95) == p_w_high
    assert pytest.approx(0.5 * 0.05) == p_s_high


def test_noma_sinr_and_rates():
    """Verify SINR and achievable rate equations against explicit test cases."""
    p_w = 0.3
    p_s = 0.1
    g_w = 1e-10
    g_s = 1e-9
    n_m = 1e-11
    b_m = 180000.0

    # Weak user SINR: gamma_w = (p_w * g_w) / (p_s * g_w + N_m)
    expected_gamma_w = (0.3 * 1e-10) / (0.1 * 1e-10 + 1e-11)  # 3e-11 / 2e-11 = 1.5
    # SIC decode SINR: gamma_{w->s} = (p_w * g_s) / (p_s * g_s + N_m)
    expected_gamma_w_s = (0.3 * 1e-9) / (0.1 * 1e-9 + 1e-11)  # 3e-10 / 1.1e-10 = 2.72727
    # Strong user SINR: gamma_s = (p_s * g_s) / N_m
    expected_gamma_s = (0.1 * 1e-9) / 1e-11  # 1e-10 / 1e-11 = 10.0

    gamma_w, gamma_w_s, gamma_s = compute_noma_sinr(p_w, p_s, g_w, g_s, n_m)
    assert pytest.approx(expected_gamma_w, rel=1e-5) == gamma_w
    assert pytest.approx(expected_gamma_w_s, rel=1e-5) == gamma_w_s
    assert pytest.approx(expected_gamma_s, rel=1e-5) == gamma_s

    # Rates:
    # R_w = B_m * min(log2(1 + 1.5), log2(1 + 2.72727)) = B_m * log2(2.5)
    # R_s = B_m * log2(1 + 10.0) = B_m * log2(11.0)
    expected_r_w = b_m * math.log2(2.5)
    expected_r_s = b_m * math.log2(11.0)

    r_w, r_s = compute_achievable_rates(gamma_w, gamma_w_s, gamma_s, b_m)
    assert pytest.approx(expected_r_w, rel=1e-5) == r_w
    assert pytest.approx(expected_r_s, rel=1e-5) == r_s


def test_frame_energy_and_normalization():
    """Verify frame energy calculation and normalization."""
    airtimes = [0.5, 0.5, 0.4, 0.6]
    pair_powers = [0.5, 0.4, 0.3, 0.5]
    t_f = 0.01
    p_max = 1.0

    # E_t = 0.01 * (0.5*0.5 + 0.5*0.4 + 0.4*0.3 + 0.6*0.5)
    # = 0.01 * (0.25 + 0.20 + 0.12 + 0.30) = 0.01 * 0.87 = 0.0087 J
    e_t, e_norm = compute_frame_energy(airtimes, pair_powers, t_f, p_max)
    assert pytest.approx(0.0087, rel=1e-5) == e_t
    assert pytest.approx(0.87, rel=1e-5) == e_norm
    assert 0.0 <= e_norm <= 1.0
