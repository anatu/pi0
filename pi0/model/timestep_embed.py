"""Sinusoidal timestep encoding and action token embedding with timestep injection."""

import math
import torch
import torch.nn as nn


def sinusoidal_encoding(tau: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal positional encoding of scalar timestep(s).

    Args:
        tau: (...,) float tensor of timestep values in [0, 1]
        dim: Embedding dimension (must be even)
    Returns:
        (..., dim) sinusoidal encoding
    """
    assert dim % 2 == 0, f"dim must be even, got {dim}"
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=tau.device, dtype=tau.dtype) / half
    )
    # (...,) x (half,) -> (..., half)
    args = tau.unsqueeze(-1) * freqs
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ActionTokenEmbedding(nn.Module):
    """Embed noisy action vectors with flow matching timestep injection.

    Implements:
        embedding = W3 · swish(W2 · concat(W1 · a, φ(τ)))

    Where φ(τ) is sinusoidal encoding of the scalar timestep τ.

    Input:  noisy actions (B, H, d_action), timestep τ scalar or (B,)
    Output: (B, H, action_expert_dim) action token embeddings
    """

    def __init__(self, action_dim: int, action_expert_dim: int, timestep_embed_dim: int):
        super().__init__()
        w = action_expert_dim
        self.timestep_embed_dim = timestep_embed_dim

        # W1: project action to action_expert_dim
        self.w1 = nn.Linear(action_dim, w)
        # W2: project concat(W1·a, φ(τ)) to 2*w
        self.w2 = nn.Linear(w + timestep_embed_dim, 2 * w)
        # W3: project back to w
        self.w3 = nn.Linear(2 * w, w)

    def forward(self, noisy_actions: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noisy_actions: (B, H, d_action) noisy action chunk
            tau: (B,) or scalar, flow matching timestep in [0, 1]
        Returns:
            (B, H, action_expert_dim)
        """
        B, H, _ = noisy_actions.shape

        # Project actions: (B, H, w)
        a_proj = self.w1(noisy_actions)

        # Timestep encoding: (B, timestep_embed_dim)
        if tau.dim() == 0:
            tau = tau.expand(B)
        phi_tau = sinusoidal_encoding(tau, self.timestep_embed_dim)  # (B, te_dim)

        # Expand timestep encoding to match action chunk: (B, H, te_dim)
        phi_tau = phi_tau.unsqueeze(1).expand(-1, H, -1)

        # Concat and transform: (B, H, w + te_dim) -> (B, H, 2w) -> (B, H, w)
        concat = torch.cat([a_proj, phi_tau], dim=-1)
        hidden = torch.nn.functional.silu(self.w2(concat))  # swish = silu
        return self.w3(hidden)
