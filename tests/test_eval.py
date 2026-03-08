"""Tests for evaluation loop, metrics computation, and baselines."""

import numpy as np
import pytest

from pi0.config import EnvConfig
from pi0.eval.evaluator import Evaluator
from pi0.eval.baselines import RandomPolicy
from pi0.env.expert_policy import ExpertPolicy


def test_evaluator_returns_valid_metrics():
    """Evaluator should return a dict with all required metric keys."""
    env_cfg = EnvConfig()
    evaluator = Evaluator(env_cfg)
    expert = ExpertPolicy(config=env_cfg)
    metrics = evaluator.evaluate(expert, num_episodes=5, seed=42)

    required_keys = {"success_rate", "avg_reward", "std_reward",
                     "avg_episode_length", "std_episode_length"}
    assert required_keys.issubset(metrics.keys()), f"Missing keys: {required_keys - metrics.keys()}"

    assert 0.0 <= metrics["success_rate"] <= 1.0
    assert isinstance(metrics["avg_reward"], float)
    assert isinstance(metrics["avg_episode_length"], float)
    assert metrics["std_reward"] >= 0.0
    assert metrics["std_episode_length"] >= 0.0


def test_expert_high_success_rate():
    """Scripted expert should have near-100% success rate."""
    env_cfg = EnvConfig()
    evaluator = Evaluator(env_cfg)
    expert = ExpertPolicy(config=env_cfg)
    metrics = evaluator.evaluate(expert, num_episodes=50, seed=100)

    assert metrics["success_rate"] >= 0.90, (
        f"Expert success rate {metrics['success_rate']:.2f} should be >= 0.90"
    )


def test_random_policy_low_success_rate():
    """Random policy should have near-zero success rate."""
    env_cfg = EnvConfig()
    evaluator = Evaluator(env_cfg)
    random_pol = RandomPolicy(config=env_cfg)
    metrics = evaluator.evaluate(random_pol, num_episodes=50, seed=200)

    assert metrics["success_rate"] <= 0.20, (
        f"Random success rate {metrics['success_rate']:.2f} should be <= 0.20"
    )


def test_random_policy_act_shape():
    """Random policy should return actions of correct shape."""
    env_cfg = EnvConfig()
    random_pol = RandomPolicy(config=env_cfg)
    obs = {"image": np.zeros((64, 64, 3), dtype=np.uint8),
           "proprio": np.zeros(4, dtype=np.float32),
           "language": "test"}
    action = random_pol.act(obs)
    assert action.shape == (env_cfg.action_dim,)
    assert action.dtype == np.float32


def test_random_policy_action_bounds():
    """Random policy actions should be within bounds."""
    env_cfg = EnvConfig()
    random_pol = RandomPolicy(config=env_cfg)
    obs = {"image": np.zeros((64, 64, 3), dtype=np.uint8),
           "proprio": np.zeros(4, dtype=np.float32),
           "language": "test"}
    for _ in range(100):
        action = random_pol.act(obs)
        assert np.all(np.abs(action) <= env_cfg.max_action)


def test_evaluator_reproducible_with_seed():
    """Same seed should give same results."""
    env_cfg = EnvConfig()
    evaluator = Evaluator(env_cfg)
    expert = ExpertPolicy(config=env_cfg)

    m1 = evaluator.evaluate(expert, num_episodes=10, seed=42)
    m2 = evaluator.evaluate(expert, num_episodes=10, seed=42)

    assert m1["success_rate"] == m2["success_rate"]
    assert m1["avg_reward"] == m2["avg_reward"]
    assert m1["avg_episode_length"] == m2["avg_episode_length"]


# ── BC-MLP Tests (slow, require CLIP) ──


@pytest.mark.slow
def test_bc_mlp_act_shape():
    """BC-MLP baseline should return actions of correct shape."""
    from pi0.eval.baselines import BCMLPBaseline

    bc = BCMLPBaseline(device="cpu")
    # Need to initialize the image encoder to test act()
    obs = {"image": np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8),
           "proprio": np.zeros(4, dtype=np.float32),
           "language": "test"}
    action = bc.act(obs)
    assert action.shape == (2,)
    assert action.dtype == np.float32
