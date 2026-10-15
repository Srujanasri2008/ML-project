"""Tests for SAC checkpoint creation, serialization, and inference loading."""

import tempfile
from pathlib import Path
import numpy as np
import pytest
from stable_baselines3 import SAC
from semantic_noma.config import ProjectConfig
from semantic_noma.environment import SemanticNomaEnv
from semantic_noma.training import train_sac


def test_checkpoint_creation_and_inference():
    """Verify SAC model trains for minimal steps, saves checkpoint, and loads for inference."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        config = ProjectConfig()
        config.training.total_timesteps = 60
        config.training.learning_starts = 10
        config.training.batch_size = 16
        config.training.eval_freq = 1000  # Do not trigger periodic eval during tiny test
        config.training.save_freq = 1000

        model, summary, run_path = train_sac(config=config, run_dir=tmp_path, verbose=0)

        # Checkpoints exist
        final_ckpt = Path(summary["final_checkpoint"])
        assert final_ckpt.exists()
        assert final_ckpt.stat().st_size > 0

        # Load checkpoint
        loaded_model = SAC.load(str(final_ckpt), device="cpu")
        env = SemanticNomaEnv(config=config)
        obs, _ = env.reset(seed=999)

        # Test deterministic inference
        action, _ = loaded_model.predict(obs, deterministic=True)
        assert action.shape == (12,)
        assert np.all(action >= -1.0) and np.all(action <= 1.0)

        # Step environment with inferred action
        next_obs, reward, terminated, truncated, info = env.step(action)
        assert next_obs.shape == (49,)
        assert np.isfinite(reward)
        assert info["power_violation"] == 0.0
        assert info["airtime_violation"] == 0.0
