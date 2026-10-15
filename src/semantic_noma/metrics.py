"""Evaluation metrics calculation and summary utilities."""

from __future__ import annotations
from dataclasses import asdict, dataclass
import math
from typing import Dict, List, Optional
import numpy as np


@dataclass
class EpisodeMetrics:
    seed: int
    policy_name: str
    episode_reward: float
    semantic_utility: float
    goodput_bps: float
    sum_rate_bps: float
    task_completion_rate: float
    deadline_miss_rate: float
    avg_latency_frames: float
    num_completed_tasks: int
    num_expired_tasks: int
    num_total_tasks: int
    energy_joules: float
    energy_efficiency_bits_per_joule: float
    fairness_index: float
    max_power_violation: float
    max_airtime_violation: float

    def to_dict(self) -> Dict[str, float | str | int]:
        return asdict(self)


def compute_jains_fairness_index(values: np.ndarray) -> float:
    """Calculate Jain's fairness index across user throughputs / service.
    
    Equation:
        J(x) = (sum_{i=1}^K x_i)^2 / (K * sum_{i=1}^K x_i^2)
        
    Returns NaN if all service values are zero.
    """
    x = np.asarray(values, dtype=np.float64)
    k = len(x)
    if k == 0:
        return float("nan")

    sum_x = float(np.sum(x))
    if sum_x <= 0:
        return float("nan")

    sum_sq = float(np.sum(x ** 2))
    if sum_sq <= 0:
        return float("nan")

    fairness = (sum_x ** 2) / (float(k) * sum_sq)
    return float(np.clip(fairness, 0.0, 1.0))


def compute_summary_statistics(
    metrics_list: List[EpisodeMetrics],
) -> Dict[str, Dict[str, float]]:
    """Compute mean, std, min, and max for each metric across episodes."""
    if not metrics_list:
        return {}

    keys = [
        "episode_reward",
        "semantic_utility",
        "goodput_bps",
        "sum_rate_bps",
        "task_completion_rate",
        "deadline_miss_rate",
        "avg_latency_frames",
        "energy_joules",
        "energy_efficiency_bits_per_joule",
        "fairness_index",
        "max_power_violation",
        "max_airtime_violation",
    ]

    summary: Dict[str, Dict[str, float]] = {}
    for key in keys:
        raw_vals = [getattr(m, key) for m in metrics_list]
        valid_vals = [v for v in raw_vals if not math.isnan(v)]
        if valid_vals:
            arr = np.array(valid_vals, dtype=np.float64)
            summary[key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
        else:
            summary[key] = {
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
            }

    return summary
