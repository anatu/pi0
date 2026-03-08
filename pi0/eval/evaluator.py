"""Run a policy in the environment and compute evaluation metrics."""

from typing import Protocol
import numpy as np
import torch
from transformers import CLIPTokenizer

from pi0.config import EnvConfig, ModelConfig, FlowConfig
from pi0.env.point_mass_env import PointMassEnv
from pi0.flow.sampler import EulerSampler


class Policy(Protocol):
    """Protocol for any policy that can be evaluated."""

    def act(self, obs: dict) -> np.ndarray: ...


class Pi0Policy:
    """Wraps a trained π0 model as a Policy for evaluation.

    Uses the EulerSampler to generate action chunks, then executes
    the first action (receding horizon control).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_config: ModelConfig,
        flow_config: FlowConfig,
        device: torch.device | str = "cpu",
    ):
        self.model = model
        self.model.eval()
        self.model_cfg = model_config
        self.sampler = EulerSampler(model, flow_config)
        self.device = torch.device(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_config.clip_model_name)

    def act(self, obs: dict) -> np.ndarray:
        """Generate an action from observation using flow matching.

        Args:
            obs: Dict with keys: image (H,W,3 uint8), proprio (4,), language (str)
        Returns:
            (d_action,) numpy action array.
        """
        # Prepare inputs as batch of 1
        image = torch.from_numpy(obs["image"]).float() / 255.0
        image = image.permute(2, 0, 1).unsqueeze(0).to(self.device)  # (1, 3, H, W)

        proprio = torch.from_numpy(obs["proprio"]).float().unsqueeze(0).to(self.device)  # (1, 4)

        encoded = self.tokenizer(
            [obs["language"]],
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        lang_tokens = encoded["input_ids"].to(self.device)  # (1, N_lang)

        # Sample action chunk via Euler integration
        action_chunk = self.sampler.sample(
            image,
            lang_tokens,
            proprio,
            action_chunk_length=self.model_cfg.action_chunk_length,
            action_dim=self.model_cfg.action_dim,
        )  # (1, H, d_action)

        # Receding horizon: take only the first action
        action = action_chunk[0, 0].cpu().numpy()
        return action


class Evaluator:
    """Evaluate a policy in the point-mass environment.

    Runs N episodes and computes success rate, average reward,
    and average episode length.
    """

    def __init__(self, env_config: EnvConfig | None = None):
        self.env_cfg = env_config or EnvConfig()

    def evaluate(
        self,
        policy: Policy,
        num_episodes: int = 100,
        seed: int = 1000,
    ) -> dict:
        """Run the policy for N episodes and return metrics.

        Args:
            policy: Any object with an act(obs) -> action method.
            num_episodes: Number of evaluation episodes.
            seed: Base seed for reproducibility.
        Returns:
            Dict with keys: success_rate, avg_reward, avg_episode_length,
                            std_reward, std_episode_length
        """
        env = PointMassEnv(self.env_cfg)

        successes = []
        rewards = []
        lengths = []

        for i in range(num_episodes):
            obs = env.reset(seed=seed + i)

            # If policy is an ExpertPolicy, set the goal and seed
            if hasattr(policy, "set_goal"):
                policy.set_goal(env.goal)
            if hasattr(policy, "seed"):
                policy.seed(seed + i)

            ep_reward = 0.0
            ep_length = 0

            for step in range(self.env_cfg.max_episode_steps):
                action = policy.act(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                ep_length += 1
                if terminated or truncated:
                    break

            successes.append(info.get("reached", False))
            rewards.append(ep_reward)
            lengths.append(ep_length)

        return {
            "success_rate": float(np.mean(successes)),
            "avg_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "avg_episode_length": float(np.mean(lengths)),
            "std_episode_length": float(np.std(lengths)),
        }
