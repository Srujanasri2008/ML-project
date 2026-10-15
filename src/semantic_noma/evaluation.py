"""Comprehensive evaluation framework for SAC and baseline policies."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from .baselines import EqualAllocationPolicy, RandomAllocationPolicy, SemanticPriorityPolicy
from .config import ProjectConfig
from .environment import SemanticNomaEnv
from .metrics import EpisodeMetrics, compute_jains_fairness_index, compute_summary_statistics


def run_episode(
    env: SemanticNomaEnv,
    policy: Any,
    seed: int,
    policy_name: str,
    record_step_details: bool = False,
) -> Tuple[EpisodeMetrics, Optional[Dict[str, Any]]]:
    """Execute a single evaluation episode deterministically."""
    obs, _ = env.reset(seed=seed)
    done = False

    ep_reward = 0.0
    sem_utils = []
    sum_rates = []
    useful_bits_total = 0.0
    user_useful_bits = np.zeros(env.config.network.num_users, dtype=np.float64)
    total_energy = 0.0
    max_p_violation = 0.0
    max_t_violation = 0.0

    step_details: Dict[str, List[Any]] = {
        "frames": [],
        "airtimes": [],
        "pair_powers": [],
        "user_rates": [],
    }

    while not done:
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        ep_reward += float(reward)
        sem_utils.append(float(info.get("semantic_utility", 0.0)))
        sum_rates.append(float(info.get("sum_rate_bps", 0.0)))
        useful_bits_total += float(info.get("useful_bits_sum", 0.0))
        total_energy += float(info.get("energy_joules", 0.0))

        user_bits = np.array(info.get("useful_bits_per_user", []), dtype=np.float64)
        if len(user_bits) == env.config.network.num_users:
            user_useful_bits += user_bits

        max_p_violation = max(max_p_violation, float(info.get("power_violation", 0.0)))
        max_t_violation = max(max_t_violation, float(info.get("airtime_violation", 0.0)))

        if record_step_details:
            step_details["frames"].append(info.get("frame", 0))
            step_details["airtimes"].append(info.get("airtimes", []))
            step_details["pair_powers"].append(info.get("pair_powers", []))

    # Calculate metrics
    sim_time_s = env.config.network.episode_length * env.config.network.frame_duration_s
    goodput_bps = useful_bits_total / sim_time_s if sim_time_s > 0 else 0.0

    completed_tasks = env.task_manager.completed_tasks
    expired_tasks = env.task_manager.expired_tasks
    num_completed = len(completed_tasks)
    num_expired = len(expired_tasks)
    total_tasks = num_completed + num_expired + len(env.task_manager.active_tasks)

    completion_rate = num_completed / float(total_tasks) if total_tasks > 0 else 0.0
    miss_rate = num_expired / float(total_tasks) if total_tasks > 0 else 0.0

    if num_completed > 0:
        latencies = [
            (t.completion_frame - t.arrival_frame + 1)
            for t in completed_tasks
            if t.completion_frame is not None
        ]
        avg_latency = float(np.mean(latencies)) if latencies else 0.0
    else:
        avg_latency = float("nan")

    energy_efficiency = (
        (useful_bits_total / total_energy) if total_energy > 0 else float("nan")
    )
    fairness = compute_jains_fairness_index(user_useful_bits)

    metrics = EpisodeMetrics(
        seed=seed,
        policy_name=policy_name,
        episode_reward=round(ep_reward, 4),
        semantic_utility=round(float(np.mean(sem_utils)), 4),
        goodput_bps=round(goodput_bps, 2),
        sum_rate_bps=round(float(np.mean(sum_rates)), 2),
        task_completion_rate=round(completion_rate, 4),
        deadline_miss_rate=round(miss_rate, 4),
        avg_latency_frames=round(avg_latency, 2),
        num_completed_tasks=num_completed,
        num_expired_tasks=num_expired,
        num_total_tasks=total_tasks,
        energy_joules=round(total_energy, 4),
        energy_efficiency_bits_per_joule=round(energy_efficiency, 2),
        fairness_index=round(fairness, 4),
        max_power_violation=round(max_p_violation, 6),
        max_airtime_violation=round(max_t_violation, 6),
    )

    return metrics, (step_details if record_step_details else None)


def evaluate_all_policies(
    config: ProjectConfig,
    sac_model_or_path: SAC | str | Path,
    test_seeds: Optional[List[int]] = None,
    output_dir: Optional[Path | str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Evaluate SAC and all baselines across held-out test seeds."""
    if test_seeds is None:
        test_seeds = config.evaluation.test_seeds

    out_path = Path(output_dir) if output_dir else Path("runs/eval")
    out_path.mkdir(parents=True, exist_ok=True)

    # Load SAC model if a path was given
    if isinstance(sac_model_or_path, (str, Path)):
        sac_model = SAC.load(str(sac_model_or_path), device="cpu")
    else:
        sac_model = sac_model_or_path

    env = SemanticNomaEnv(config=config)

    policies = {
        "SAC": sac_model,
        "Equal Allocation": EqualAllocationPolicy(config=config),
        "Semantic Priority": SemanticPriorityPolicy(config=config),
        "Random Allocation": RandomAllocationPolicy(seed=config.training.seed, config=config),
    }

    all_episode_metrics: List[EpisodeMetrics] = []
    example_allocation_traces: Dict[str, Any] = {}

    for pol_name, policy in policies.items():
        for i, seed in enumerate(test_seeds):
            # Record step details for first seed to visualize allocations
            record_steps = (i == 0)
            metrics, step_details = run_episode(
                env=env,
                policy=policy,
                seed=seed,
                policy_name=pol_name,
                record_step_details=record_steps,
            )
            all_episode_metrics.append(metrics)
            if record_steps and step_details is not None:
                example_allocation_traces[pol_name] = step_details

    # Convert to DataFrames
    per_seed_df = pd.DataFrame([m.to_dict() for m in all_episode_metrics])
    per_seed_df.to_csv(out_path / "per_seed_metrics.csv", index=False)

    # Build summary table across policies
    summary_rows = []
    for pol_name in policies.keys():
        sub_metrics = [m for m in all_episode_metrics if m.policy_name == pol_name]
        stats = compute_summary_statistics(sub_metrics)
        row = {"policy": pol_name}
        for metric_name, vals in stats.items():
            row[f"{metric_name}_mean"] = vals["mean"]
            row[f"{metric_name}_std"] = vals["std"]
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_path / "evaluation_summary.csv", index=False)

    return per_seed_df, summary_df, example_allocation_traces
