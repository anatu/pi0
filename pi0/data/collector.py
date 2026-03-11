"""Collect expert demonstration trajectories."""

from pathlib import Path
import numpy as np
from tqdm import tqdm

from pi0.config import EnvConfig, DataConfig
from pi0.env.point_mass_env import PointMassEnv
from pi0.env.expert_policy import ExpertPolicy
from pi0.data.storage import save_trajectory


def collect_trajectories(
    num_trajectories: int,
    output_dir: str | Path,
    env_config: EnvConfig | None = None,
    expert_noise_std: float = 0.002,
    seed: int = 0,
    show_progress: bool = True,
) -> None:
    """Run the expert policy and save trajectories to disk.

    Args:
        num_trajectories: Number of episodes to collect.
        output_dir: Directory to write .npz files.
        env_config: Environment configuration.
        expert_noise_std: Gaussian noise added to expert actions.
        seed: Base random seed.
        show_progress: Show tqdm progress bar.
    """
    cfg = env_config or EnvConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = PointMassEnv(cfg)
    expert = ExpertPolicy(config=cfg, noise_std=expert_noise_std)

    iterator = range(num_trajectories)
    if show_progress:
        iterator = tqdm(iterator, desc="Collecting trajectories")

    for i in iterator:
        ep_seed = seed + i
        obs = env.reset(seed=ep_seed)
        expert.seed(ep_seed)
        expert.set_goal(env.goal)

        images, proprios, actions = [], [], []
        images.append(obs["image"])
        proprios.append(obs["proprio"])

        for _ in range(cfg.max_episode_steps):
            action = expert.act(obs)
            actions.append(action)
            obs, _, terminated, truncated, _ = env.step(action)
            images.append(obs["image"])
            proprios.append(obs["proprio"])
            if terminated or truncated:
                break

        # images has T+1 entries, proprios has T+1, actions has T
        # Align: keep first T images/proprios to match T actions
        T = len(actions)
        images_arr = np.stack(images[:T], axis=0)
        proprios_arr = np.stack(proprios[:T], axis=0)
        actions_arr = np.stack(actions, axis=0)

        save_trajectory(
            output_dir / f"traj_{i:05d}.npz",
            images=images_arr,
            proprio=proprios_arr,
            actions=actions_arr,
            language=cfg.language_command,
        )
