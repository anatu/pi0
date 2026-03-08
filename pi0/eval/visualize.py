"""Record policy rollouts and save as GIF visualizations."""

from pathlib import Path

import numpy as np
import imageio

from pi0.config import EnvConfig
from pi0.env.point_mass_env import PointMassEnv


def record_episode(
    env: PointMassEnv,
    policy,
    seed: int = 0,
) -> tuple[list[np.ndarray], dict]:
    """Run one episode and collect rendered frames.

    Args:
        env: The environment instance.
        policy: Any object with act(obs) -> action.
        seed: Seed for this episode.
    Returns:
        (frames, info) where frames is a list of (H, W, 3) uint8 arrays
        and info contains episode metrics.
    """
    obs = env.reset(seed=seed)
    if hasattr(policy, "set_goal"):
        policy.set_goal(env.goal)
    if hasattr(policy, "seed"):
        policy.seed(seed)

    frames = [obs["image"].copy()]
    ep_reward = 0.0

    for _ in range(env.cfg.max_episode_steps):
        action = policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(obs["image"].copy())
        ep_reward += reward
        if terminated or truncated:
            break

    episode_info = {
        "reward": ep_reward,
        "length": len(frames) - 1,
        "reached": info.get("reached", False),
    }
    return frames, episode_info


def save_gif(
    frames: list[np.ndarray],
    path: str | Path,
    fps: int = 15,
    scale: int = 4,
) -> None:
    """Save a list of frames as a GIF file.

    Args:
        frames: List of (H, W, 3) uint8 arrays.
        path: Output file path (should end in .gif).
        fps: Frames per second.
        scale: Upscale factor for better visibility.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if scale > 1:
        scaled = []
        for f in frames:
            scaled.append(
                np.repeat(np.repeat(f, scale, axis=0), scale, axis=1)
            )
        frames = scaled

    imageio.mimsave(str(path), frames, fps=fps, loop=0)


def record_and_save_episodes(
    policy,
    output_path: str | Path,
    num_episodes: int = 5,
    env_config: EnvConfig | None = None,
    seed: int = 2000,
    fps: int = 15,
    scale: int = 4,
) -> list[dict]:
    """Record multiple episodes and save as a single GIF (concatenated).

    Creates one GIF with all episodes played sequentially, separated
    by a brief pause (white frames).

    Args:
        policy: Policy with act(obs) -> action.
        output_path: Path to save the GIF.
        num_episodes: Number of episodes to record.
        env_config: Environment config.
        seed: Base seed.
        fps: Frames per second.
        scale: Upscale factor.
    Returns:
        List of episode info dicts.
    """
    cfg = env_config or EnvConfig()
    env = PointMassEnv(cfg)

    all_frames = []
    all_info = []

    for i in range(num_episodes):
        frames, info = record_episode(env, policy, seed=seed + i)
        all_frames.extend(frames)
        all_info.append(info)

        # Add separator frames (white) between episodes
        if i < num_episodes - 1:
            h, w = frames[0].shape[:2]
            separator = np.full((h, w, 3), 240, dtype=np.uint8)
            all_frames.extend([separator] * 5)

    save_gif(all_frames, output_path, fps=fps, scale=scale)
    return all_info
