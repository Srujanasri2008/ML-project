"""Baseline resource allocation policies: Equal, Semantic-Priority, and Random."""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from .config import ProjectConfig


class EqualAllocationPolicy:
    """Baseline A: Equal airtime (tau_g = 0.5), full power (eta_g = 1.0), fixed split (beta_g = 0.75).
    
    Produces raw action [-1, 1]^12 that yields exact equal allocation:
        eta_g = 1.0 -> a_eta = 1.0
        beta_g = 0.75 -> a_beta = (0.75 - 0.5)/0.45 * 2 - 1 = 1/9 ≈ 0.111111
        z_g = 1.0 -> a_tau = 1.0 -> tau_g = 1 / max(1, 1+1) = 0.5
    """

    def __init__(self, config: Optional[ProjectConfig] = None):
        self.config = config if config is not None else ProjectConfig()
        num_pairs = self.config.network.num_pairs

        # Compute raw actions corresponding to target values
        a_eta = 1.0
        a_beta = float((0.75 - 0.5) / 0.45 * 2.0 - 1.0)  # ≈ 0.111111
        a_tau = 1.0

        action = []
        for _ in range(num_pairs):
            action.extend([a_eta, a_beta, a_tau])
        self.fixed_action = np.array(action, dtype=np.float32)

    def predict(
        self,
        obs: np.ndarray,
        state: Optional[np.ndarray] = None,
        episode_start: Optional[bool] = None,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return the fixed equal-allocation raw action."""
        return self.fixed_action.copy(), None


class SemanticPriorityPolicy:
    """Baseline B: Deterministic priority heuristic based on semantic importance and deadlines.
    
    Priority formula for user i:
        psi_i = (w_i / (remaining_deadline_ratio + 0.1)) * (remaining_demand_ratio + 0.1)
        
    For pair g = (u_a, u_b):
        Psi_g = psi_{u_a} + psi_{u_b}
        
    Airtime preference:
        z_g proportional to Psi_g
        
    Power utilization:
        eta_g = clip(Psi_g / max_Psi, 0.4, 1.0)
        
    Power split beta_g:
        Higher power to weak user if weak user has higher priority or demand.
    """

    def __init__(self, config: Optional[ProjectConfig] = None):
        self.config = config if config is not None else ProjectConfig()

    def predict(
        self,
        obs: np.ndarray,
        state: Optional[np.ndarray] = None,
        episode_start: Optional[bool] = None,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Compute heuristic raw actions from current observation vector."""
        num_users = self.config.network.num_users
        num_pairs = self.config.network.num_pairs

        # Extract features for each user from 49-dim obs:
        # Obs has 6 features per user:
        # 0: log_gain, 1: norm_dist, 2: rem_demand_ratio, 3: norm_demand, 4: weight, 5: rem_deadline_ratio
        user_priorities = np.zeros(num_users, dtype=np.float64)
        for u in range(num_users):
            offset = u * 6
            rem_demand_ratio = float(obs[offset + 2])
            weight = float(obs[offset + 4])
            rem_deadline_ratio = float(obs[offset + 5])

            urgency = 1.0 / (rem_deadline_ratio + 0.1)
            user_priorities[u] = weight * urgency * (rem_demand_ratio + 0.1)

        raw_action = np.zeros(num_pairs * 3, dtype=np.float32)

        # Compute pair priorities
        pair_priorities = np.zeros(num_pairs, dtype=np.float64)
        for g in range(num_pairs):
            u_a, u_b = self.config.network.pair_user_mapping[g]
            pair_priorities[g] = user_priorities[u_a] + user_priorities[u_b]

        max_pri = max(1e-6, float(np.max(pair_priorities)))

        for g in range(num_pairs):
            u_a, u_b = self.config.network.pair_user_mapping[g]
            pri_a = user_priorities[u_a]
            pri_b = user_priorities[u_b]

            # Eta: scaled power utilization in [0.4, 1.0]
            eta_val = float(np.clip(pair_priorities[g] / max_pri, 0.4, 1.0))
            a_eta = 2.0 * eta_val - 1.0

            # Beta: power fraction to weak user.
            # In NOMA, weak user needs higher power (>= 0.5).
            # If weak user has higher priority, allocate up to 0.9; else around 0.6.
            # Compare channel gains from obs (offset + 0)
            g_a = float(obs[u_a * 6])
            g_b = float(obs[u_b * 6])
            pri_weak = pri_a if g_a <= g_b else pri_b
            pri_strong = pri_b if g_a <= g_b else pri_a

            weak_bias = pri_weak / (pri_weak + pri_strong + 1e-6)
            beta_val = float(0.5 + 0.45 * np.clip(weak_bias, 0.2, 0.95))
            a_beta = float((beta_val - 0.5) / 0.45 * 2.0 - 1.0)

            # Airtime preference z_g in [0.2, 1.0]
            z_val = float(np.clip(pair_priorities[g] / max_pri, 0.2, 1.0))
            a_tau = 2.0 * z_val - 1.0

            raw_action[3 * g] = float(np.clip(a_eta, -1.0, 1.0))
            raw_action[3 * g + 1] = float(np.clip(a_beta, -1.0, 1.0))
            raw_action[3 * g + 2] = float(np.clip(a_tau, -1.0, 1.0))

        return raw_action, None


class RandomAllocationPolicy:
    """Baseline C: Random feasible allocation sampled uniformly in [-1, 1]^12 with independent RNG."""

    def __init__(self, seed: Optional[int] = 42, config: Optional[ProjectConfig] = None):
        self.config = config if config is not None else ProjectConfig()
        self.rng = np.random.default_rng(seed)
        self.action_dim = self.config.network.num_pairs * 3

    def predict(
        self,
        obs: np.ndarray,
        state: Optional[np.ndarray] = None,
        episode_start: Optional[bool] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Sample random continuous action in [-1, 1]^12."""
        action = self.rng.uniform(-1.0, 1.0, size=self.action_dim).astype(np.float32)
        return action, None
