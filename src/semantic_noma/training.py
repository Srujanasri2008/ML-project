"""SAC Agent training execution with resource limits and checkpoint management."""

from __future__ import annotations
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList

from .callbacks import MetricsLoggingCallback, PeriodicValidationCallback, TimeLimitCallback
from .config import ProjectConfig
from .environment import SemanticNomaEnv


def setup_thread_limits(max_cpu_threads: int = 2) -> None:
    """Enforce strict thread limit for PyTorch and numerical libraries."""
    torch.set_num_threads(max_cpu_threads)
    os.environ["OMP_NUM_THREADS"] = str(max_cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(max_cpu_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(max_cpu_threads)


def train_sac(
    config: ProjectConfig,
    run_dir: Optional[Path | str] = None,
    verbose: int = 1,
) -> Tuple[SAC, Dict[str, Any], Path]:
    """Train the SAC model within strict resource and time limits.
    
    Returns:
        (trained_model, run_summary, run_dir_path)
    """
    setup_thread_limits(config.training.max_cpu_threads)

    if run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_path = Path("runs") / f"run_{timestamp}"
    else:
        run_path = Path(run_dir)

    run_path.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = run_path / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Save run config
    config.save(run_path / "run_config.json")

    # Instantiate training environment
    train_env = SemanticNomaEnv(config=config)

    # Policy network architecture
    policy_kwargs = {
        "net_arch": config.training.hidden_layers,
    }

    # SAC model instantiation
    model = SAC(
        policy=config.training.policy_type,
        env=train_env,
        learning_rate=config.training.learning_rate,
        buffer_size=config.training.buffer_size,
        learning_starts=config.training.learning_starts,
        batch_size=config.training.batch_size,
        gamma=config.training.gamma,
        train_freq=config.training.train_freq,
        gradient_steps=config.training.gradient_steps,
        policy_kwargs=policy_kwargs,
        seed=config.training.seed,
        device="cpu",
        verbose=0,
    )

    # Setup callbacks
    time_limit_cb = TimeLimitCallback(
        max_runtime_seconds=config.training.max_runtime_seconds, verbose=verbose
    )
    logging_cb = MetricsLoggingCallback(
        log_file=run_path / "training_log.csv", verbose=verbose
    )
    val_cb = PeriodicValidationCallback(
        eval_env_factory=lambda: SemanticNomaEnv(config=config),
        val_seeds=config.evaluation.validation_seeds,
        eval_freq=config.training.eval_freq,
        checkpoint_dir=checkpoints_dir,
        verbose=verbose,
    )

    callbacks = CallbackList([time_limit_cb, logging_cb, val_cb])

    start_time = time.time()
    try:
        model.learn(
            total_timesteps=config.training.total_timesteps,
            callback=callbacks,
            log_interval=10,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Preserving logs and checkpoints.")

    elapsed_time = time.time() - start_time
    timesteps_completed = model.num_timesteps

    # Save final model
    final_model_path = checkpoints_dir / "final_model.zip"
    model.save(str(final_model_path))

    # Determine best model path
    best_model_path = checkpoints_dir / "best_model.zip"
    active_eval_path = best_model_path if best_model_path.exists() else final_model_path

    run_summary: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "run_directory": str(run_path.resolve()),
        "timesteps_requested": config.training.total_timesteps,
        "timesteps_completed": timesteps_completed,
        "elapsed_seconds": round(elapsed_time, 2),
        "wall_clock_limit_reached": time_limit_cb.time_limit_reached,
        "best_checkpoint": str(active_eval_path.resolve()),
        "final_checkpoint": str(final_model_path.resolve()),
    }

    return model, run_summary, run_path
