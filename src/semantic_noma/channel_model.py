"""Wireless channel model with log-distance path loss and Rayleigh fading."""

from __future__ import annotations
import math
import numpy as np
from typing import Optional, Sequence
from .config import NetworkConfig


def compute_free_space_path_loss(
    carrier_frequency_hz: float,
    reference_distance_m: float = 1.0,
    speed_of_light_m_s: float = 3.0e8,
) -> float:
    """Calculate free-space path loss PL_0 at reference distance d_0 (in dB).
    
    PL_0 = 20 * log10((4 * pi * d_0 * f_c) / c)
    """
    numerator = 4.0 * math.pi * reference_distance_m * carrier_frequency_hz
    val = numerator / speed_of_light_m_s
    return 20.0 * math.log10(val)


def compute_path_loss(
    distance_m: float,
    pl_0_db: float,
    path_loss_exponent: float = 3.5,
    reference_distance_m: float = 1.0,
) -> float:
    """Calculate path loss for user at distance d_i (in dB).
    
    PL_i = PL_0 + 10 * n * log10(d_i / d_0)
    """
    if distance_m <= 0:
        raise ValueError("Distance must be strictly positive.")
    ratio = distance_m / reference_distance_m
    return pl_0_db + 10.0 * path_loss_exponent * math.log10(ratio)


class ChannelModel:
    """Simulates wireless channel power gains for all users across frames.
    
    Uses log-distance path loss and independent complex Gaussian (Rayleigh) fading.
    g_i(t) = 10^(-PL_i / 10) * |h_i(t)|^2
    where h_i(t) ~ CN(0, 1), so E[|h_i(t)|^2] = 1.
    """

    def __init__(
        self,
        config: NetworkConfig,
        user_distances: Optional[Sequence[float]] = None,
        seed: Optional[int] = 42,
    ):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.pl_0 = compute_free_space_path_loss(
            config.carrier_frequency_hz,
            config.reference_distance_m,
            config.speed_of_light_m_s,
        )

        if user_distances is not None:
            if len(user_distances) != config.num_users:
                raise ValueError(
                    f"Expected {config.num_users} distances, got {len(user_distances)}"
                )
            self.distances = np.array(user_distances, dtype=np.float64)
        else:
            # Deterministically space users across configured range if no distances given
            d_min, d_max = config.user_distance_range_m
            self.distances = np.linspace(d_min, d_max, config.num_users)

        # Precompute large-scale path loss and channel attenuation
        self.path_losses_db = np.array(
            [
                compute_path_loss(
                    d, self.pl_0, config.path_loss_exponent, config.reference_distance_m
                )
                for d in self.distances
            ],
            dtype=np.float64,
        )
        self.large_scale_gains = 10.0 ** (-self.path_losses_db / 10.0)

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the channel fading generator with an optional seed."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def sample_fading_power(self) -> np.ndarray:
        """Sample complex Gaussian fading coefficients and compute power |h|^2.
        
        For CN(0, 1), real and imag parts ~ N(0, 0.5), so |h|^2 ~ Exp(1).
        """
        # Complex normal CN(0, 1) has real ~ N(0, 0.5) and imag ~ N(0, 0.5)
        # Power |h|^2 is exponentially distributed with mean 1
        return self.rng.exponential(scale=1.0, size=self.config.num_users)

    def step(self) -> np.ndarray:
        """Generate instantaneous channel power gains g_i(t) for all users."""
        fading_power = self.sample_fading_power()
        gains = self.large_scale_gains * fading_power
        # Ensure non-negative and non-NaN
        return np.maximum(gains, 1e-25)
