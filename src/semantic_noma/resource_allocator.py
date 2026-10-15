"""Resource allocation engine for Hybrid NOMA-OFDMA."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from .config import NetworkConfig
from .wireless_math import (
    compute_achievable_rates,
    compute_frame_energy,
    compute_noma_sinr,
    compute_noise_power,
    decompose_pair_power,
)


@dataclass
class AllocationResult:
    eta: np.ndarray             # Pair power utilization fraction (num_pairs,)
    beta: np.ndarray            # Weak user power fraction (num_pairs,)
    airtimes: np.ndarray        # Pair airtime fraction tau_g (num_pairs,)
    pair_powers: np.ndarray     # Pair transmit power P_g in Watts (num_pairs,)
    user_powers: np.ndarray     # Transmit power per user when active (num_users,)
    user_rates: np.ndarray      # Achievable rate per user in bits/s (num_users,)
    service_bits: np.ndarray    # Delivered bits c_i(t) = tau_g * T_f * R_i(t) (num_users,)
    weak_users: List[int]       # Weak user index for each pair
    strong_users: List[int]     # Strong user index for each pair
    energy_joules: float        # Frame energy E_t
    energy_normalized: float    # E_norm in [0, 1]
    power_violation: float      # Any excess power > P_max (should be 0.0)
    airtime_violation: float    # Any excess airtime > 1.0 (should be 0.0)


class ResourceAllocator:
    """Computes power splits, airtime, SINR, rates, and service bits from raw actions."""

    def __init__(self, config: NetworkConfig):
        self.config = config
        self.noise_power_w = compute_noise_power(
            config.thermal_noise_density_dbm_hz,
            config.subchannel_bandwidth_hz,
            config.noise_figure_db,
        )

    def transform_raw_actions(
        self, raw_action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert raw continuous SAC actions [-1, 1]^12 into feasible (eta, beta, tau).
        
        Action layout (3 per pair, 4 pairs = 12 elements):
            For pair g:
                raw_action[3*g + 0] -> a_{eta, g}
                raw_action[3*g + 1] -> a_{beta, g}
                raw_action[3*g + 2] -> a_{tau, g}
        """
        raw = np.clip(np.asarray(raw_action, dtype=np.float64), -1.0, 1.0)
        if raw.shape[0] != self.config.num_pairs * 3:
            raise ValueError(
                f"Expected action of length {self.config.num_pairs * 3}, got {raw.shape[0]}"
            )

        eta = np.zeros(self.config.num_pairs, dtype=np.float64)
        beta = np.zeros(self.config.num_pairs, dtype=np.float64)
        z = np.zeros(self.config.num_pairs, dtype=np.float64)
        tau = np.zeros(self.config.num_pairs, dtype=np.float64)

        for g in range(self.config.num_pairs):
            a_eta = raw[3 * g]
            a_beta = raw[3 * g + 1]
            a_tau = raw[3 * g + 2]

            # Equation 6.3 / 8.2: eta_g in [0, 1]
            eta[g] = (a_eta + 1.0) / 2.0
            # beta_g in [0.5, 0.95]
            beta[g] = 0.5 + 0.45 * ((a_beta + 1.0) / 2.0)
            # Airtime preference z_g in [0, 1]
            z[g] = (a_tau + 1.0) / 2.0

        # Subchannel-wise normalization for airtime: tau_g = z_g / max(1, sum_{j in G_m} z_j)
        for m, pairs in self.config.subchannel_pair_mapping.items():
            subchannel_sum = float(np.sum(z[pairs]))
            normalizer = max(1.0, subchannel_sum)
            for g in pairs:
                tau[g] = z[g] / normalizer

        return eta, beta, tau

    def allocate(
        self,
        raw_action: np.ndarray,
        channel_gains: np.ndarray,
    ) -> AllocationResult:
        """Perform full allocation, SINR, rate, and service computation."""
        eta, beta, tau = self.transform_raw_actions(raw_action)

        num_pairs = self.config.num_pairs
        num_users = self.config.num_users
        num_subchannels = self.config.num_subchannels
        p_max = self.config.p_max_w
        t_f = self.config.frame_duration_s
        b_m = self.config.subchannel_bandwidth_hz

        pair_powers = np.zeros(num_pairs, dtype=np.float64)
        user_powers = np.zeros(num_users, dtype=np.float64)
        user_rates = np.zeros(num_users, dtype=np.float64)
        service_bits = np.zeros(num_users, dtype=np.float64)
        weak_users: List[int] = []
        strong_users: List[int] = []

        # Find weak and strong users for each pair based on current channel gains
        for g in range(num_pairs):
            u_a, u_b = self.config.pair_user_mapping[g]
            g_a = float(channel_gains[u_a])
            g_b = float(channel_gains[u_b])

            if g_a <= g_b:
                u_w, u_s = u_a, u_b
                g_w, g_s = g_a, g_b
            else:
                u_w, u_s = u_b, u_a
                g_w, g_s = g_b, g_a

            weak_users.append(u_w)
            strong_users.append(u_s)

            # Decompose pair power
            p_g, p_w, p_s = decompose_pair_power(eta[g], beta[g], p_max, num_subchannels)
            pair_powers[g] = p_g
            user_powers[u_w] = p_w
            user_powers[u_s] = p_s

            # Calculate NOMA SINR
            gamma_w, gamma_w_s, gamma_s = compute_noma_sinr(
                p_w, p_s, g_w, g_s, self.noise_power_w
            )

            # Calculate achievable rates
            r_w, r_s = compute_achievable_rates(gamma_w, gamma_w_s, gamma_s, b_m)
            user_rates[u_w] = r_w
            user_rates[u_s] = r_s

            # Calculate service bits in this frame: c_i(t) = tau_g * T_f * R_i(t)
            service_bits[u_w] = tau[g] * t_f * r_w
            service_bits[u_s] = tau[g] * t_f * r_s

        # Energy calculation
        e_t, e_norm = compute_frame_energy(tau, pair_powers, t_f, p_max)

        # Constraint violation verification
        # 1. Total instantaneous power on any active combination across subchannels <= P_max
        # Since each subchannel has at most 1 active pair with power <= P_max/M,
        # max instantaneous power across M subchannels is sum_m (P_max/M) = P_max.
        # Check actual pair powers:
        power_violation = 0.0
        for m, pairs in self.config.subchannel_pair_mapping.items():
            max_pair_p = max(pair_powers[p] for p in pairs)
            if max_pair_p > (p_max / num_subchannels) + 1e-9:
                power_violation += max_pair_p - (p_max / num_subchannels)

        # 2. Airtime per subchannel sum <= 1.0
        airtime_violation = 0.0
        for m, pairs in self.config.subchannel_pair_mapping.items():
            sum_tau = float(np.sum(tau[pairs]))
            if sum_tau > 1.0 + 1e-9:
                airtime_violation += sum_tau - 1.0

        return AllocationResult(
            eta=eta,
            beta=beta,
            airtimes=tau,
            pair_powers=pair_powers,
            user_powers=user_powers,
            user_rates=user_rates,
            service_bits=service_bits,
            weak_users=weak_users,
            strong_users=strong_users,
            energy_joules=e_t,
            energy_normalized=e_norm,
            power_violation=float(power_violation),
            airtime_violation=float(airtime_violation),
        )
