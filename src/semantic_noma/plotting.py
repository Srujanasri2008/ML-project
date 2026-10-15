"""Plotting module for training diagnostics and comparative evaluation."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive background rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_curves(training_log_csv: Path | str, output_dir: Path | str) -> None:
    """Plot training reward and reward components over timesteps."""
    csv_path = Path(training_log_csv)
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if df.empty or "timestep" not in df.columns:
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Training reward vs timesteps
    plt.figure(figsize=(8, 5))
    plt.plot(df["timestep"], df["episode_reward"], color="#1f77b4", lw=2, label="Episode Reward")
    if len(df) > 5:
        # Rolling average
        rolling = df["episode_reward"].rolling(window=min(10, len(df)), min_periods=1).mean()
        plt.plot(df["timestep"], rolling, color="#ff7f0e", lw=2.5, label="Rolling Mean")
    plt.xlabel("Training Timesteps", fontsize=12)
    plt.ylabel("Cumulative Episode Reward", fontsize=12)
    plt.title("SAC Training Reward Curve", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(out / "training_reward.png", dpi=200)
    plt.close()

    # 2. Reward components
    plt.figure(figsize=(9, 5))
    if "mean_semantic_utility" in df.columns:
        plt.plot(df["timestep"], df["mean_semantic_utility"], label="Semantic Utility U_sem", color="#2ca02c", lw=2)
    if "mean_energy_penalty" in df.columns:
        plt.plot(df["timestep"], df["mean_energy_penalty"], label="Energy Penalty (lambda_E * E_norm)", color="#d62728", lw=1.8, linestyle="--")
    if "mean_deadline_penalty" in df.columns:
        plt.plot(df["timestep"], df["mean_deadline_penalty"], label="Deadline Penalty (lambda_D * M_t)", color="#9467bd", lw=1.8, linestyle=":")
    plt.xlabel("Training Timesteps", fontsize=12)
    plt.ylabel("Reward Component Value", fontsize=12)
    plt.title("Training Reward Decomposition", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(out / "reward_components.png", dpi=200)
    plt.close()


def plot_comparative_evaluations(
    summary_csv: Path | str,
    per_seed_csv: Path | str,
    output_dir: Path | str,
) -> None:
    """Generate bar charts with error bars for comparative metrics."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary_path = Path(summary_csv)
    if not summary_path.exists():
        return
    df = pd.read_csv(summary_path)
    if df.empty or "policy" not in df.columns:
        return

    policies = df["policy"].tolist()
    colors = ["#2b5c8f", "#e26d5c", "#38b000", "#7209b7"][: len(policies)]

    # Helper function for bar charts
    def _plot_metric_bar(mean_col: str, std_col: str, title: str, ylabel: str, filename: str):
        if mean_col not in df.columns:
            return
        means = df[mean_col].tolist()
        stds = df[std_col].tolist() if std_col in df.columns else [0.0] * len(means)

        plt.figure(figsize=(8, 5))
        bars = plt.bar(policies, means, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor="black")
        plt.title(title, fontsize=13, fontweight="bold")
        plt.ylabel(ylabel, fontsize=11)
        plt.grid(axis="y", linestyle="--", alpha=0.6)

        # Annotate bar values
        for bar, m in zip(bars, means):
            y_pos = bar.get_height()
            if not np.isnan(y_pos):
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    y_pos + 0.02 * (max(means) if max(means) > 0 else 1.0),
                    f"{m:.3g}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        plt.tight_layout()
        plt.savefig(out / filename, dpi=200)
        plt.close()

    # 3. Semantic utility
    _plot_metric_bar(
        "semantic_utility_mean",
        "semantic_utility_std",
        "Mean Semantic Utility Comparison",
        "Semantic Utility U_sem (per frame)",
        "semantic_utility_comparison.png",
    )

    # 4. Goodput & sum rate
    if "goodput_bps_mean" in df.columns:
        plt.figure(figsize=(9, 5))
        x = np.arange(len(policies))
        w = 0.35
        plt.bar(x - w / 2, df["goodput_bps_mean"] / 1e3, w, label="Goodput (kbps)", color="#1f77b4", alpha=0.85, edgecolor="black")
        if "sum_rate_bps_mean" in df.columns:
            plt.bar(x + w / 2, df["sum_rate_bps_mean"] / 1e3, w, label="Sum Rate (kbps)", color="#ff7f0e", alpha=0.85, edgecolor="black")
        plt.xticks(x, policies, fontsize=11)
        plt.ylabel("Data Rate (kbps)", fontsize=11)
        plt.title("Throughput & Goodput Comparison", fontsize=13, fontweight="bold")
        plt.legend(fontsize=11)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(out / "throughput_comparison.png", dpi=200)
        plt.close()

    # 5. Latency & Deadline miss
    if "avg_latency_frames_mean" in df.columns and "deadline_miss_rate_mean" in df.columns:
        fig, ax1 = plt.subplots(figsize=(9, 5))
        x = np.arange(len(policies))
        w = 0.35

        ax1.bar(x - w / 2, df["avg_latency_frames_mean"], w, color="#2ca02c", label="Avg Latency (frames)", alpha=0.85, edgecolor="black")
        ax1.set_ylabel("Average Latency (frames)", color="#2ca02c", fontsize=11)
        ax1.set_xticks(x)
        ax1.set_xticklabels(policies, fontsize=11)

        ax2 = ax1.twinx()
        ax2.bar(x + w / 2, df["deadline_miss_rate_mean"] * 100.0, w, color="#d62728", label="Miss Rate (%)", alpha=0.85, edgecolor="black")
        ax2.set_ylabel("Deadline Miss Rate (%)", color="#d62728", fontsize=11)

        plt.title("Task Latency & Deadline-Miss Rate", fontsize=13, fontweight="bold")
        fig.tight_layout()
        plt.savefig(out / "latency_comparison.png", dpi=200)
        plt.close()

    # 6. Energy efficiency
    _plot_metric_bar(
        "energy_efficiency_bits_per_joule_mean",
        "energy_efficiency_bits_per_joule_std",
        "Energy Efficiency Comparison",
        "Energy Efficiency (bits / Joule)",
        "energy_efficiency_comparison.png",
    )

    # 7. Jain's Fairness Index
    _plot_metric_bar(
        "fairness_index_mean",
        "fairness_index_std",
        "User Service Fairness (Jain's Index)",
        "Jain's Fairness Index",
        "fairness_comparison.png",
    )


def plot_example_allocations(
    allocation_traces: Dict[str, Any], output_dir: Path | str
) -> None:
    """Plot power and airtime allocations over episode frames for an example run."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    trace = allocation_traces.get("SAC") or next(iter(allocation_traces.values()), None)
    if not trace or "frames" not in trace:
        return

    frames = trace["frames"]
    pair_powers = np.array(trace["pair_powers"])  # shape (T, num_pairs)
    airtimes = np.array(trace["airtimes"])        # shape (T, num_pairs)

    num_pairs = pair_powers.shape[1] if len(pair_powers.shape) > 1 else 0

    # 8. Power allocation example
    if num_pairs > 0:
        plt.figure(figsize=(10, 5))
        for g in range(num_pairs):
            plt.plot(frames, pair_powers[:, g], label=f"Pair {g} Power", lw=1.8)
        plt.xlabel("Transmission Frame", fontsize=11)
        plt.ylabel("Transmit Power (Watts)", fontsize=11)
        plt.title("Example Dynamic Power Allocation Across Pairs (SAC)", fontsize=13, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(out / "power_allocation_example.png", dpi=200)
        plt.close()

    # 9. Airtime allocation example
    if num_pairs > 0:
        plt.figure(figsize=(10, 5))
        for g in range(num_pairs):
            plt.plot(frames, airtimes[:, g], label=f"Pair {g} Airtime tau_{g}", lw=1.8)
        plt.xlabel("Transmission Frame", fontsize=11)
        plt.ylabel("Airtime Fraction tau", fontsize=11)
        plt.title("Example Dynamic Airtime Allocation Across Candidate Pairs (SAC)", fontsize=13, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(out / "airtime_allocation_example.png", dpi=200)
        plt.close()
