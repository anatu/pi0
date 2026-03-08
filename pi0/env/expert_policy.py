"""Scripted proportional-control expert policy for the point-mass env."""

import numpy as np
from pi0.config import EnvConfig


class ExpertPolicy:
    """P-controller that drives toward the goal with optional noise."""

    def __init__(self, gain: float = 0.5, noise_std: float = 0.002, config: EnvConfig | None = None):
        self.gain = gain
        self.noise_std = noise_std
        self.cfg = config or EnvConfig()
        self.rng = np.random.default_rng()

    def seed(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def act(self, obs: dict) -> np.ndarray:
        proprio = obs["proprio"]
        pos = proprio[:2]
        vel = proprio[2:4]
        goal = self._goal

        # Desired velocity toward goal, damped
        desired_vel = self.gain * (goal - pos)
        action = desired_vel - vel
        action = np.clip(action, -self.cfg.max_action, self.cfg.max_action)

        if self.noise_std > 0:
            action = action + self.rng.normal(0, self.noise_std, size=action.shape)
            action = np.clip(action, -self.cfg.max_action, self.cfg.max_action)

        return action.astype(np.float32)

    def set_goal(self, goal: np.ndarray):
        """Set the goal the expert should reach (called externally with env's goal)."""
        self._goal = np.asarray(goal, dtype=np.float32)
