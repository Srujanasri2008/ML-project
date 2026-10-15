"""Configuration dataclasses and loaders for Semantic NOMA-OFDMA."""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass
class NetworkConfig:
    num_users: int = 8
    num_pairs: int = 4
    num_subchannels: int = 2
    pairs_per_subchannel: int = 2
    p_max_w: float = 1.0
    subchannel_bandwidth_hz: float = 180000.0
    frame_duration_s: float = 0.01
    episode_length: int = 50
    carrier_frequency_hz: float = 3.5e9
    path_loss_exponent: float = 3.5
    reference_distance_m: float = 1.0
    speed_of_light_m_s: float = 3.0e8
    user_distance_range_m: Tuple[float, float] = (30.0, 250.0)
    noise_figure_db: float = 7.0
    thermal_noise_density_dbm_hz: float = -174.0
    subchannel_pair_mapping: Dict[int, List[int]] = field(
        default_factory=lambda: {0: [0, 1], 1: [2, 3]}
    )
    pair_user_mapping: Dict[int, List[int]] = field(
        default_factory=lambda: {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7]}
    )


@dataclass
class TaskConfig:
    demand_range_bits: Tuple[int, int] = (2000, 10000)
    deadline_range_frames: Tuple[int, int] = (2, 6)
    importance_range: Tuple[float, float] = (0.2, 1.0)


@dataclass
class RewardConfig:
    lambda_energy: float = 0.01
    lambda_deadline: float = 0.5


@dataclass
class TrainingConfig:
    algorithm: str = "SAC"
    policy_type: str = "MlpPolicy"
    hidden_layers: List[int] = field(default_factory=lambda: [128, 128])
    learning_rate: float = 0.0003
    gamma: float = 0.99
    buffer_size: int = 25000
    learning_starts: int = 500
    batch_size: int = 128
    train_freq: int = 4
    gradient_steps: int = 1
    total_timesteps: int = 20000
    max_runtime_seconds: int = 600
    max_cpu_threads: int = 2
    seed: int = 42
    eval_freq: int = 2000
    eval_episodes: int = 5
    save_freq: int = 5000


@dataclass
class EvaluationConfig:
    validation_seeds: List[int] = field(
        default_factory=lambda: [1000, 1001, 1002, 1003, 1004]
    )
    test_seeds: List[int] = field(
        default_factory=lambda: [
            2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
            2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019
        ]
    )


@dataclass
class ProjectConfig:
    network: NetworkConfig = field(default_factory=NetworkConfig)
    tasks: TaskConfig = field(default_factory=TaskConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProjectConfig:
        net_data = data.get("network", {})
        # Convert subchannel and pair mappings keys to int
        if "subchannel_pair_mapping" in net_data:
            net_data["subchannel_pair_mapping"] = {
                int(k): v for k, v in net_data["subchannel_pair_mapping"].items()
            }
        if "pair_user_mapping" in net_data:
            net_data["pair_user_mapping"] = {
                int(k): v for k, v in net_data["pair_user_mapping"].items()
            }
        if "user_distance_range_m" in net_data:
            net_data["user_distance_range_m"] = tuple(net_data["user_distance_range_m"])

        task_data = data.get("tasks", {})
        if "demand_range_bits" in task_data:
            task_data["demand_range_bits"] = tuple(task_data["demand_range_bits"])
        if "deadline_range_frames" in task_data:
            task_data["deadline_range_frames"] = tuple(task_data["deadline_range_frames"])
        if "importance_range" in task_data:
            task_data["importance_range"] = tuple(task_data["importance_range"])

        reward_data = data.get("reward", {})
        train_data = data.get("training", {})
        eval_data = data.get("evaluation", {})

        return cls(
            network=NetworkConfig(**net_data),
            tasks=TaskConfig(**task_data),
            reward=RewardConfig(**reward_data),
            training=TrainingConfig(**train_data),
            evaluation=EvaluationConfig(**eval_data),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> ProjectConfig:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "network": {
                "num_users": self.network.num_users,
                "num_pairs": self.network.num_pairs,
                "num_subchannels": self.network.num_subchannels,
                "pairs_per_subchannel": self.network.pairs_per_subchannel,
                "p_max_w": self.network.p_max_w,
                "subchannel_bandwidth_hz": self.network.subchannel_bandwidth_hz,
                "frame_duration_s": self.network.frame_duration_s,
                "episode_length": self.network.episode_length,
                "carrier_frequency_hz": self.network.carrier_frequency_hz,
                "path_loss_exponent": self.network.path_loss_exponent,
                "reference_distance_m": self.network.reference_distance_m,
                "speed_of_light_m_s": self.network.speed_of_light_m_s,
                "user_distance_range_m": list(self.network.user_distance_range_m),
                "noise_figure_db": self.network.noise_figure_db,
                "thermal_noise_density_dbm_hz": self.network.thermal_noise_density_dbm_hz,
                "subchannel_pair_mapping": {
                    str(k): v for k, v in self.network.subchannel_pair_mapping.items()
                },
                "pair_user_mapping": {
                    str(k): v for k, v in self.network.pair_user_mapping.items()
                },
            },
            "tasks": {
                "demand_range_bits": list(self.tasks.demand_range_bits),
                "deadline_range_frames": list(self.tasks.deadline_range_frames),
                "importance_range": list(self.tasks.importance_range),
            },
            "reward": {
                "lambda_energy": self.reward.lambda_energy,
                "lambda_deadline": self.reward.lambda_deadline,
            },
            "training": {
                "algorithm": self.training.algorithm,
                "policy_type": self.training.policy_type,
                "hidden_layers": self.training.hidden_layers,
                "learning_rate": self.training.learning_rate,
                "gamma": self.training.gamma,
                "buffer_size": self.training.buffer_size,
                "learning_starts": self.training.learning_starts,
                "batch_size": self.training.batch_size,
                "train_freq": self.training.train_freq,
                "gradient_steps": self.training.gradient_steps,
                "total_timesteps": self.training.total_timesteps,
                "max_runtime_seconds": self.training.max_runtime_seconds,
                "max_cpu_threads": self.training.max_cpu_threads,
                "seed": self.training.seed,
                "eval_freq": self.training.eval_freq,
                "eval_episodes": self.training.eval_episodes,
                "save_freq": self.training.save_freq,
            },
            "evaluation": {
                "validation_seeds": self.evaluation.validation_seeds,
                "test_seeds": self.evaluation.test_seeds,
            },
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
