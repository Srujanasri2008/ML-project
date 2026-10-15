"""Tests for SemanticNomaEnv compliance with Gymnasium and dimensional specifications."""

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest
from semantic_noma.config import ProjectConfig
from semantic_noma.environment import SemanticNomaEnv


def test_observation_and_action_dimensions():
    """Verify state has exactly 49 features and action has exactly 12 continuous values."""
    env = SemanticNomaEnv()
    assert env.observation_space.shape == (49,)
    assert env.action_space.shape == (12,)

    obs, info = env.reset(seed=42)
    assert obs.shape == (49,)
    assert np.all(np.isfinite(obs))
    assert env.observation_space.contains(obs)


def test_gymnasium_env_checker():
    """Run standard Gymnasium environment checker to verify API compliance."""
    env = SemanticNomaEnv()
    # check_env raises exceptions if environment violates standard Gymnasium contracts
    check_env(env.unwrapped)


def test_episode_truncation_and_reset():
    """Verify episode truncates at exactly 50 frames and reset restores state."""
    config = ProjectConfig()
    config.network.episode_length = 50
    env = SemanticNomaEnv(config=config)

    obs, _ = env.reset(seed=100)
    for frame in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)
        assert not terminated
        if frame < 49:
            assert not truncated
        else:
            assert truncated

    # Reset
    obs_new, _ = env.reset(seed=100)
    assert env.current_frame == 0
    assert obs_new.shape == (49,)


def test_reward_decomposition():
    """Verify total reward equals U_sem - lambda_E * E_norm - lambda_D * M_t."""
    env = SemanticNomaEnv()
    env.reset(seed=42)
    action = env.action_space.sample()
    _, reward, _, _, info = env.step(action)

    u_sem = info["semantic_utility"]
    e_pen = info["reward_energy_penalty"]
    d_pen = info["reward_deadline_penalty"]
    expected_reward = u_sem - e_pen - d_pen

    assert pytest.approx(expected_reward, abs=1e-6) == reward
