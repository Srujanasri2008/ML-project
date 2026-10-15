"""Mathematical equations for wireless transmission, NOMA SINR, rates, and energy."""

from __future__ import annotations
import math
from typing import Tuple
import numpy as np


def compute_noise_power(
    thermal_noise_density_dbm_hz: float,
    bandwidth_hz: float,
    noise_figure_db: float,
) -> float:
    """Convert thermal noise density and noise figure to noise power N_m (in Watts).
    
    Equation:
        N_m = 10^((thermal_noise_density_dbm_hz - 30) / 10) * B_m * 10^(noise_figure_db / 10)
    """
    noise_density_w_hz = 10.0 ** ((thermal_noise_density_dbm_hz - 30.0) / 10.0)
    noise_factor = 10.0 ** (noise_figure_db / 10.0)
    noise_power = noise_density_w_hz * bandwidth_hz * noise_factor
    return float(noise_power)


def decompose_pair_power(
    eta_g: float,
    beta_g: float,
    p_max_w: float,
    num_subchannels: int,
) -> Tuple[float, float, float]:
    """Decompose candidate pair power into weak and strong user powers.
    
    Equations:
        P_g = eta_g * (P_max / M)
        p_{w, g} = beta_g * P_g
        p_{s, g} = (1 - beta_g) * P_g
        
    Enforces:
        0 <= eta_g <= 1
        0.5 <= beta_g <= 0.95
    """
    eta_clipped = float(np.clip(eta_g, 0.0, 1.0))
    beta_clipped = float(np.clip(beta_g, 0.5, 0.95))

    p_sub_max = p_max_w / float(num_subchannels)
    p_g = eta_clipped * p_sub_max
    p_w = beta_clipped * p_g
    p_s = (1.0 - beta_clipped) * p_g

    return p_g, p_w, p_s


def compute_noma_sinr(
    p_w: float,
    p_s: float,
    g_w: float,
    g_s: float,
    noise_power_w: float,
) -> Tuple[float, float, float]:
    """Compute downlink NOMA SINR for weak user, SIC decode, and strong user.
    
    Equations:
        gamma_w = (p_w * g_w) / (p_s * g_w + N_m)
        gamma_{w -> s} = (p_w * g_s) / (p_s * g_s + N_m)
        gamma_s = (p_s * g_s) / N_m
        
    Returns:
        (gamma_w, gamma_w_s, gamma_s)
    """
    if noise_power_w <= 0:
        raise ValueError("Noise power must be strictly positive.")

    # Guard against negative or NaN inputs
    p_w = max(0.0, float(p_w))
    p_s = max(0.0, float(p_s))
    g_w = max(1e-30, float(g_w))
    g_s = max(1e-30, float(g_s))

    denom_w = p_s * g_w + noise_power_w
    gamma_w = (p_w * g_w) / denom_w if denom_w > 0 else 0.0

    denom_w_s = p_s * g_s + noise_power_w
    gamma_w_s = (p_w * g_s) / denom_w_s if denom_w_s > 0 else 0.0

    gamma_s = (p_s * g_s) / noise_power_w

    return float(gamma_w), float(gamma_w_s), float(gamma_s)


def compute_achievable_rates(
    gamma_w: float,
    gamma_w_s: float,
    gamma_s: float,
    bandwidth_hz: float,
) -> Tuple[float, float]:
    """Calculate achievable data rates in bits/s.
    
    Equations:
        R_w = B_m * min(log2(1 + gamma_w), log2(1 + gamma_{w -> s}))
        R_s = B_m * log2(1 + gamma_s)
    """
    gamma_w = max(0.0, float(gamma_w))
    gamma_w_s = max(0.0, float(gamma_w_s))
    gamma_s = max(0.0, float(gamma_s))

    rate_w_cand1 = bandwidth_hz * math.log2(1.0 + gamma_w)
    rate_w_cand2 = bandwidth_hz * math.log2(1.0 + gamma_w_s)
    r_w = min(rate_w_cand1, rate_w_cand2)

    r_s = bandwidth_hz * math.log2(1.0 + gamma_s)

    return float(r_w), float(r_s)


def compute_frame_energy(
    airtimes: np.ndarray | Sequence[float],
    pair_powers: np.ndarray | Sequence[float],
    frame_duration_s: float,
    p_max_w: float,
) -> Tuple[float, float]:
    """Calculate absolute and normalized transmission energy for one frame.
    
    Equations:
        E_t = T_f * sum_g (tau_g * P_g)
        E_norm = E_t / (P_max * T_f)
    """
    tau = np.asarray(airtimes, dtype=np.float64)
    p = np.asarray(pair_powers, dtype=np.float64)

    e_t = float(frame_duration_s * np.sum(tau * p))
    e_max = float(p_max_w * frame_duration_s)
    e_norm = e_t / e_max if e_max > 0 else 0.0

    return e_t, float(np.clip(e_norm, 0.0, 1.0))
