"""Tests for the point-mass environment and expert policy."""

import numpy as np
from pi0.config import EnvConfig
from pi0.env.point_mass_env import PointMassEnv
from pi0.env.expert_policy import ExpertPolicy


def _make_env():
    return PointMassEnv(EnvConfig())


def test_reset_returns_correct_shapes():
    env = _make_env()
    obs = env.reset(seed=42)
    assert obs["image"].shape == (64, 64, 3)
    assert obs["image"].dtype == np.uint8
    assert obs["proprio"].shape == (4,)
    assert obs["proprio"].dtype == np.float32
    assert isinstance(obs["language"], str)


def test_step_returns_correct_shapes():
    env = _make_env()
    env.reset(seed=42)
    action = np.array([0.01, 0.01], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs["image"].shape == (64, 64, 3)
    assert obs["proprio"].shape == (4,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "dist_to_goal" in info


def test_reward_is_negative_distance():
    env = _make_env()
    env.reset(seed=42)
    action = np.zeros(2, dtype=np.float32)
    _, reward, _, _, info = env.step(action)
    # Reward should be -dist (possibly + bonus if reached)
    assert reward <= 0 or info["reached"]


def test_episode_truncates_at_max_steps():
    env = _make_env()
    env.reset(seed=42)
    for _ in range(100):
        _, _, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
        if terminated or truncated:
            break
    assert truncated or terminated


def test_termination_on_reach():
    env = _make_env()
    env.reset(seed=42)
    # Teleport agent to goal
    env.pos = env.goal.copy()
    env.vel = np.zeros(2, dtype=np.float32)
    _, _, terminated, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert info["reached"]
    assert terminated


def test_expert_reaches_goal():
    cfg = EnvConfig()
    env = PointMassEnv(cfg)
    expert = ExpertPolicy(config=cfg)
    successes = 0
    n_trials = 100
    for i in range(n_trials):
        obs = env.reset(seed=i)
        expert.set_goal(env.goal)
        for _ in range(cfg.max_episode_steps):
            action = expert.act(obs)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated:
                successes += 1
                break
            if truncated:
                break
    assert successes / n_trials >= 0.90, f"Expert success rate {successes}/{n_trials}"


def test_image_pixel_range():
    env = _make_env()
    obs = env.reset(seed=42)
    img = obs["image"]
    assert img.min() >= 0
    assert img.max() <= 255
