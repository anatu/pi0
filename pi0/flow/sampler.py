"""Euler integration sampler for generating actions from noise."""

import torch
import torch.nn as nn

from pi0.config import FlowConfig


class EulerSampler:
    """Generate action chunks by integrating the learned velocity field.

    Starting from A_0 ~ N(0, I) (pure noise), iteratively apply:
        A_{τ+δ} = A_τ + δ * v_θ(A_τ, o)

    with K steps from τ=0 to τ=1 (δ = 1/K).
    """

    def __init__(self, model: nn.Module, config: FlowConfig):
        self.model = model
        self.config = config
        self.K = config.euler_steps
        self.delta = 1.0 / self.K

    @torch.no_grad()
    def sample(
        self,
        images: torch.Tensor,
        language_tokens: torch.Tensor,
        proprio: torch.Tensor,
        action_chunk_length: int,
        action_dim: int,
    ) -> torch.Tensor:
        """Generate an action chunk from noise via Euler integration.

        Args:
            images: (B, 3, H, W) image observations.
            language_tokens: (B, N_lang) language token IDs.
            proprio: (B, d_proprio) proprioceptive state.
            action_chunk_length: H, number of actions in the chunk.
            action_dim: Dimension of each action.
        Returns:
            (B, H, action_dim) generated action chunk.
        """
        B = images.shape[0]
        device = images.device

        # Start from pure noise
        actions = torch.randn(B, action_chunk_length, action_dim, device=device)

        # Euler integration: K steps from τ=0 to τ=1
        for k in range(self.K):
            tau_value = k * self.delta
            tau = torch.full((B,), tau_value, device=device)

            # Predict velocity field
            velocity = self.model(images, language_tokens, proprio, actions, tau)

            # Euler step
            actions = actions + self.delta * velocity

        return actions
