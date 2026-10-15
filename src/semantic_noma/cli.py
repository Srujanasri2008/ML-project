"""Command-line interface and main entrypoint runner."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Optional

from .config import ProjectConfig
from .evaluation import evaluate_all_policies
from .plotting import plot_comparative_evaluations, plot_example_allocations, plot_training_curves
from .training import train_sac


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Soft Actor-Critic Semantic-Aware NOMA-OFDMA Resource Allocation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON configuration file (defaults to configs/smoke_test.json if --smoke else configs/default.json)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run fast smoke-test training (1,000 timesteps) to verify pipeline",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total training timesteps",
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=int,
        default=None,
        help="Override maximum training wall-clock runtime in seconds",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help="Number of held-out evaluation test episodes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for training",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to an existing model checkpoint to evaluate without retraining",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and run evaluation on checkpoint",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run pytest automated test suite before starting experiment",
    )
    return parser.parse_args(args)


def run_experiment(args: argparse.Namespace) -> Path:
    # 1. Automated tests if requested
    if args.run_tests:
        print("\n=== Running Automated Unit Tests ===")
        test_result = subprocess.run([sys.executable, "-m", "pytest", "-q"], capture_output=False)
        if test_result.returncode != 0:
            print("Tests failed. Please resolve test issues before running training.")
            sys.exit(test_result.returncode)

    # 2. Load Configuration
    if args.config:
        config_path = Path(args.config)
    elif args.smoke:
        config_path = Path("configs/smoke_test.json")
    else:
        config_path = Path("configs/default.json")

    print(f"\n[Config] Loading configuration from {config_path}")
    config = ProjectConfig.from_file(config_path)

    # Apply CLI overrides
    if args.timesteps is not None:
        config.training.total_timesteps = args.timesteps
    if args.max_runtime_seconds is not None:
        config.training.max_runtime_seconds = args.max_runtime_seconds
    if args.seed is not None:
        config.training.seed = args.seed
    if args.eval_episodes is not None:
        # Generate requested number of test seeds starting at 2000
        config.evaluation.test_seeds = [2000 + i for i in range(args.eval_episodes)]

    print(f"[Config] Target Training Timesteps: {config.training.total_timesteps}")
    print(f"[Config] Max Runtime: {config.training.max_runtime_seconds} seconds")
    print(f"[Config] Test Seeds: {len(config.evaluation.test_seeds)} scenarios")

    # 3. Training / Checkpoint resolution
    model_checkpoint_path: Path
    if args.skip_train or (args.checkpoint is not None and not args.smoke):
        if not args.checkpoint:
            raise ValueError("--checkpoint must be specified when using --skip-train")
        model_checkpoint_path = Path(args.checkpoint)
        if not model_checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {model_checkpoint_path}")
        run_path = model_checkpoint_path.parent.parent
        print(f"\n[Mode] Skipping training; using checkpoint: {model_checkpoint_path}")
        run_summary = {
            "mode": "eval_only",
            "checkpoint": str(model_checkpoint_path.resolve()),
        }
    else:
        print("\n=== Stage: Training SAC Policy ===")
        model, run_summary, run_path = train_sac(config=config, verbose=1)
        model_checkpoint_path = Path(run_summary["best_checkpoint"])
        print(f"\n[Training Complete] Timesteps completed: {run_summary['timesteps_completed']}")
        print(f"[Checkpoint] Using checkpoint: {model_checkpoint_path}")

    # Plot training curves if training log exists
    train_log_file = run_path / "training_log.csv"
    if train_log_file.exists():
        print("[Plotting] Generating training curve plots...")
        plot_training_curves(training_log_csv=train_log_file, output_dir=run_path)

    # 4. Evaluation against Baselines
    print("\n=== Stage: Evaluating SAC and Baselines on Held-Out Test Scenarios ===")
    per_seed_df, summary_df, alloc_traces = evaluate_all_policies(
        config=config,
        sac_model_or_path=model_checkpoint_path,
        test_seeds=config.evaluation.test_seeds,
        output_dir=run_path,
    )
    print("\n[Evaluation Summary]")
    cols_to_show = [
        c for c in summary_df.columns
        if c in ["policy", "semantic_utility_mean", "goodput_bps_mean", "avg_latency_frames_mean", "deadline_miss_rate_mean", "fairness_index_mean"]
    ]
    print(summary_df[cols_to_show].to_string(index=False))

    # 5. Comparative and Allocation Plots
    print("\n[Plotting] Generating evaluation bar charts and allocation graphs...")
    plot_comparative_evaluations(
        summary_csv=run_path / "evaluation_summary.csv",
        per_seed_csv=run_path / "per_seed_metrics.csv",
        output_dir=run_path,
    )
    plot_example_allocations(allocation_traces=alloc_traces, output_dir=run_path)

    # 6. Save final run summary JSON
    run_summary["evaluation_test_seeds"] = config.evaluation.test_seeds
    run_summary["evaluation_summary"] = summary_df.to_dict(orient="records")
    with open(run_path / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    print(f"\n=== Experiment Finished Successfully ===")
    print(f"All outputs, plots, and CSVs saved in: {run_path.resolve()}\n")
    return run_path


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
