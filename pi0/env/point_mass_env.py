"""2D point-mass reaching environment with image rendering."""

import numpy as np
from PIL import Image, ImageDraw
from pi0.config import EnvConfig


class PointMassEnv:
    """A 2D point-mass must reach a colored target.

    Observations:
        image: (H, W, 3) uint8 RGB top-down view
        proprio: [x, y, vx, vy] float32
        language: str

    Actions:
        [dvx, dvy] float32, clipped to [-max_action, max_action]
    """

    def __init__(self, config: EnvConfig | None = None):
        self.cfg = config or EnvConfig()
        self.rng = np.random.default_rng()
        self.pos = np.zeros(2, dtype=np.float32)
        self.vel = np.zeros(2, dtype=np.float32)
        self.goal = np.zeros(2, dtype=np.float32)
        self.step_count = 0

    def seed(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            self.seed(seed)
        margin = 0.1
        lo = self.cfg.workspace_min + margin
        hi = self.cfg.workspace_max - margin
        self.pos = self.rng.uniform(lo, hi, size=2).astype(np.float32)
        self.vel = np.zeros(2, dtype=np.float32)
        self.goal = self.rng.uniform(lo, hi, size=2).astype(np.float32)
        # Ensure goal is not too close to start
        while np.linalg.norm(self.goal - self.pos) < 0.15:
            self.goal = self.rng.uniform(lo, hi, size=2).astype(np.float32)
        self.step_count = 0
        return self._get_obs()

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -self.cfg.max_action, self.cfg.max_action)

        self.vel = self.vel + action
        speed = np.linalg.norm(self.vel)
        if speed > self.cfg.max_speed:
            self.vel = self.vel * (self.cfg.max_speed / speed)

        self.pos = self.pos + self.vel * self.cfg.dt
        self.pos = np.clip(self.pos, self.cfg.workspace_min, self.cfg.workspace_max)

        self.step_count += 1

        dist = np.linalg.norm(self.pos - self.goal)
        reward = -dist
        reached = dist < self.cfg.reach_threshold
        if reached:
            reward += self.cfg.reach_bonus

        terminated = bool(reached)
        truncated = bool(self.step_count >= self.cfg.max_episode_steps)

        info = {"dist_to_goal": dist, "reached": reached}
        return self._get_obs(), float(reward), terminated, truncated, info

    def _get_obs(self) -> dict:
        return {
            "image": self._render(),
            "proprio": np.array(
                [self.pos[0], self.pos[1], self.vel[0], self.vel[1]],
                dtype=np.float32,
            ),
            "language": self.cfg.language_command,
        }

    def _render(self) -> np.ndarray:
        sz = self.cfg.image_size
        img = Image.new("RGB", (sz, sz), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        def to_pixel(p):
            rng = self.cfg.workspace_max - self.cfg.workspace_min
            x = int((p[0] - self.cfg.workspace_min) / rng * (sz - 1))
            y = int((1.0 - (p[1] - self.cfg.workspace_min) / rng) * (sz - 1))
            return x, y

        # Draw target (red circle)
        gx, gy = to_pixel(self.goal)
        r = max(3, sz // 16)
        draw.ellipse([gx - r, gy - r, gx + r, gy + r], fill=(220, 40, 40))

        # Draw agent (blue circle)
        ax, ay = to_pixel(self.pos)
        r_agent = max(2, sz // 20)
        draw.ellipse(
            [ax - r_agent, ay - r_agent, ax + r_agent, ay + r_agent],
            fill=(40, 40, 220),
        )

        return np.array(img, dtype=np.uint8)

    @property
    def action_space_low(self) -> np.ndarray:
        return np.full(self.cfg.action_dim, -self.cfg.max_action, dtype=np.float32)

    @property
    def action_space_high(self) -> np.ndarray:
        return np.full(self.cfg.action_dim, self.cfg.max_action, dtype=np.float32)
