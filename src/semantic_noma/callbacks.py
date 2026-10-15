"""Training callbacks for wall-clock limits, logging, and validation checkpointing."""

from __future__ import annotations
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

from .config import ProjectConfig


class TimeLimitCallback(BaseCallback):
    """Gracefully halts training if wall-clock budget is exceeded."""

    def __init__(self, max_runtime_seconds: int, verbose: int = 0):
        super().__init__(verbose)
        self.max_runtime_seconds = max_runtime_seconds
        self.start_time: float = 0.0
        self.time_limit_reached = False

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        self.time_limit_reached = False

    def _on_step(self) -> bool:
        elapsed = time.time() - self.start_time
        if elapsed >= self.max_runtime_seconds:
            if self.verbose > 0:
                print(
                    f"\n[TimeLimitCallback] Reached wall-clock limit of {self.max_runtime_seconds}s "
                    f"(elapsed: {elapsed:.1f}s). Stopping training gracefully."
                )
            self.time_limit_reached = True
            return False
        return True


class MetricsLoggingCallback(BaseCallback):
    """Logs episode-level metrics, rewards, penalties, and wall-clock times to CSV."""

    def __init__(self, log_file: Path | str, verbose: int = 0):
        super().__init__(verbose)
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.records: List[Dict[str, Any]] = []
        self.start_time: float = 0.0

        # Accumulators for current episode
        self.curr_reward = 0.0
        self.curr_sem_util = 0.0
        self.curr_energy_pen = 0.0
        self.curr_deadline_pen = 0.0
        self.curr_useful_bits = 0.0
        self.ep_steps = 0

    def _on_training_start(self) -> None:
        self.start_time = time.time()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [0.0])

        self.curr_reward += float(rewards[0])
        self.ep_steps += 1

        if infos and len(infos) > 0:
            info = infos[0]
            self.curr_sem_util += float(info.get("semantic_utility", 0.0))
            self.curr_energy_pen += float(info.get("reward_energy_penalty", 0.0))
            self.curr_deadline_pen += float(info.get("reward_deadline_penalty", 0.0))
            self.curr_useful_bits += float(info.get("useful_bits_sum", 0.0))

            # Check if episode ended (truncated or terminated)
            dones = self.locals.get("dones", [False])
            if dones[0]:
                elapsed = time.time() - self.start_time
                rec = {
                    "timestep": self.num_timesteps,
                    "elapsed_seconds": round(elapsed, 2),
                    "episode_reward": round(self.curr_reward, 4),
                    "mean_semantic_utility": round(self.curr_sem_util / max(1, self.ep_steps), 4),
                    "mean_energy_penalty": round(self.curr_energy_pen / max(1, self.ep_steps), 4),
                    "mean_deadline_penalty": round(self.curr_deadline_pen / max(1, self.ep_steps), 4),
                    "total_useful_bits": round(self.curr_useful_bits, 1),
                    "total_completed_tasks": info.get("total_completed", 0),
                    "total_expired_tasks": info.get("total_expired", 0),
                }
                self.records.append(rec)
                self.save_csv()

                # Reset accumulators
                self.curr_reward = 0.0
                self.curr_sem_util = 0.0
                self.curr_energy_pen = 0.0
                self.curr_deadline_pen = 0.0
                self.curr_useful_bits = 0.0
                self.ep_steps = 0

        return True

    def save_csv(self) -> None:
        if self.records:
            df = pd.DataFrame(self.records)
            df.to_csv(self.log_file, index=False)


class PeriodicValidationCallback(BaseCallback):
    """Evaluates policy on fixed validation seeds and checkpoints best model."""

    def __init__(
        self,
        eval_env_factory,
        val_seeds: List[int],
        eval_freq: int,
        checkpoint_dir: Path | str,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env_factory = eval_env_factory
        self.val_seeds = val_seeds
        self.eval_freq = eval_freq
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_mean_reward = -float("inf")
        self.best_model_path = self.checkpoint_dir / "best_model.zip"

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0 and self.num_timesteps > 0:
            env = self.eval_env_factory()
            val_rewards = []

            for seed in self.val_seeds:
                obs, _ = env.reset(seed=seed)
                done = False
                ep_rew = 0.0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    ep_rew += reward
                    done = terminated or truncated
                val_rewards.append(ep_rew)

            mean_val_reward = float(np.mean(val_rewards))
            if self.verbose > 0:
                print(
                    f"\n[Validation @ step {self.num_timesteps}] Mean Reward: {mean_val_reward:.4f} "
                    f"across {len(self.val_seeds)} seeds."
                )

            # Save best validation model
            if mean_val_reward > self.best_mean_reward:
                self.best_mean_reward = mean_val_reward
                self.model.save(str(self.best_model_path))
                if self.verbose > 0:
                    print(f"  --> New best model saved to {self.best_model_path}")

            # Also save step checkpoint
            step_ckpt = self.checkpoint_dir / f"checkpoint_{self.num_timesteps}.zip"
            self.model.save(str(step_ckpt))

        return True
