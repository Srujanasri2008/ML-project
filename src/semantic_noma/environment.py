"""Gymnasium environment for Semantic-Aware Hybrid NOMA-OFDMA."""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .channel_model import ChannelModel
from .config import ProjectConfig
from .resource_allocator import AllocationResult, ResourceAllocator
from .task_model import TaskManager


class SemanticNomaEnv(gym.Env):
    """Continuous control environment for Semantic NOMA-OFDMA resource allocation.
    
    Observation space (49 dimensions):
        For each of the 8 users:
            1. Normalized log channel gain
            2. Normalized user distance
            3. Remaining-demand ratio (q_i / D_i)
            4. Original task demand normalized by max demand (D_i / D_max)
            5. Semantic importance weight w_i
            6. Remaining deadline normalized by max deadline (deadline / dl_max)
        Plus 1 global feature:
            49. Remaining fraction of current episode: (episode_length - frame) / episode_length
            
    Action space (12 dimensions):
        Box(-1.0, 1.0, shape=(12,))
        For each pair g in {0, 1, 2, 3}:
            [a_{eta, g}, a_{beta, g}, a_{tau, g}]
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[ProjectConfig] = None):
        super().__init__()
        self.config = config if config is not None else ProjectConfig()

        num_users = self.config.network.num_users
        num_pairs = self.config.network.num_pairs

        # Exactly 49 observation features
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(num_users * 6 + 1,),
            dtype=np.float32,
        )

        # Exactly 12 action dimensions
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(num_pairs * 3,),
            dtype=np.float32,
        )

        self.channel_model = ChannelModel(self.config.network, seed=self.config.training.seed)
        self.task_manager = TaskManager(
            num_users=num_users,
            config=self.config.tasks,
            seed=self.config.training.seed,
        )
        self.allocator = ResourceAllocator(self.config.network)

        self.current_frame = 0
        self.current_channels = np.zeros(num_users, dtype=np.float64)

    def _get_obs(self) -> np.ndarray:
        """Construct exactly 49 normalized observation features."""
        num_users = self.config.network.num_users
        d_min, d_max = self.config.network.user_distance_range_m
        d_range = max(1.0, d_max - d_min)
        _, d_max_bits = self.config.tasks.demand_range_bits
        _, dl_max_frames = self.config.tasks.deadline_range_frames

        features: list[float] = []

        for user_id in range(num_users):
            # 1. Log channel gain normalized
            # Channel gains typically ~ 10^-14 to 10^-6 -> log10 is -14 to -6
            g_i = max(1e-25, float(self.current_channels[user_id]))
            log_g = (np.log10(g_i) + 10.0) / 5.0
            features.append(float(np.clip(log_g, -3.0, 3.0)))

            # 2. Normalized distance
            dist = self.channel_model.distances[user_id]
            norm_dist = (dist - d_min) / d_range
            features.append(float(np.clip(norm_dist, 0.0, 1.0)))

            task = self.task_manager.active_tasks[user_id]
            # 3. Remaining-demand ratio
            features.append(float(task.remaining_demand_ratio))

            # 4. Normalized original demand
            norm_demand = task.original_demand_bits / float(d_max_bits)
            features.append(float(np.clip(norm_demand, 0.0, 1.0)))

            # 5. Semantic importance weight
            features.append(float(np.clip(task.importance_weight, 0.0, 1.0)))

            # 6. Normalized remaining deadline
            norm_deadline = task.remaining_deadline_frames / float(dl_max_frames)
            features.append(float(np.clip(norm_deadline, 0.0, 1.0)))

        # 49th feature: Remaining fraction of current episode
        rem_episode_frac = (self.config.network.episode_length - self.current_frame) / float(
            self.config.network.episode_length
        )
        features.append(float(np.clip(rem_episode_frac, 0.0, 1.0)))

        obs = np.array(features, dtype=np.float32)
        assert obs.shape == (49,), f"Observation must have exactly 49 elements, got {obs.shape}"
        return obs

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment state, channel model, and task manager."""
        super().reset(seed=seed)

        if seed is not None:
            self.channel_model.reset(seed=seed)
            self.task_manager.reset(seed=seed + 1000)
        else:
            self.task_manager.reset()

        self.current_frame = 0
        self.current_channels = self.channel_model.step()

        obs = self._get_obs()
        info = {
            "frame": self.current_frame,
            "active_tasks": len(self.task_manager.active_tasks),
        }
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one transmission frame, update tasks, compute reward, and advance state."""
        # 1-2. Perform allocation and compute delivered service bits
        alloc_res: AllocationResult = self.allocator.allocate(
            raw_action=action, channel_gains=self.current_channels
        )

        # 3-8. Process tasks in strict sequential order
        (
            useful_bits,
            user_progress,
            mean_sem_utility,
            newly_completed,
            newly_expired,
        ) = self.task_manager.process_frame(
            service_bits=alloc_res.service_bits,
            current_frame=self.current_frame,
        )

        # Compute deadline penalty M_t
        # M_t = (1 / K) * sum_{i in D_t} w_i * (remaining_q / D_i)
        num_users = self.config.network.num_users
        if newly_expired:
            missed_sum = sum(
                task.importance_weight * rem_frac for task, rem_frac in newly_expired
            )
            m_t = float(missed_sum / num_users)
        else:
            m_t = 0.0

        # Reward equation: r_t = U_sem(t) - lambda_E * E_norm(t) - lambda_D * M_t
        lambda_e = self.config.reward.lambda_energy
        lambda_d = self.config.reward.lambda_deadline
        energy_penalty = float(lambda_e * alloc_res.energy_normalized)
        deadline_penalty = float(lambda_d * m_t)
        reward = float(mean_sem_utility - energy_penalty - deadline_penalty)

        # Advance frame
        self.current_frame += 1
        terminated = False
        truncated = bool(self.current_frame >= self.config.network.episode_length)

        # Step channel gains for next frame if not finished
        if not truncated:
            self.current_channels = self.channel_model.step()

        obs = self._get_obs()

        info: Dict[str, Any] = {
            "frame": self.current_frame,
            "semantic_utility": mean_sem_utility,
            "normalized_energy": alloc_res.energy_normalized,
            "energy_joules": alloc_res.energy_joules,
            "deadline_penalty_metric": m_t,
            "reward_energy_penalty": energy_penalty,
            "reward_deadline_penalty": deadline_penalty,
            "reward_raw": reward,
            "useful_bits_sum": float(np.sum(useful_bits)),
            "useful_bits_per_user": useful_bits.tolist(),
            "sum_rate_bps": float(np.sum(alloc_res.user_rates)),
            "power_violation": alloc_res.power_violation,
            "airtime_violation": alloc_res.airtime_violation,
            "newly_completed_count": len(newly_completed),
            "newly_expired_count": len(newly_expired),
            "total_completed": len(self.task_manager.completed_tasks),
            "total_expired": len(self.task_manager.expired_tasks),
            "airtimes": alloc_res.airtimes.tolist(),
            "pair_powers": alloc_res.pair_powers.tolist(),
        }

        return obs, reward, terminated, truncated, info
