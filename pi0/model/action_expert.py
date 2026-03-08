"""Action expert (Expert 2): FFN layers for proprioceptive and action tokens."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionExpertFFN(nn.Module):
    """Feed-forward network for Expert 2 (action expert).

    Same structure as BackboneFFN but at the smaller action expert width.
    Processes proprio and action tokens.
    """

    def __init__(self, action_expert_dim: int, mlp_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(action_expert_dim)
        self.fc1 = nn.Linear(action_expert_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, action_expert_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N_action, action_expert_dim)
        Returns:
            (B, N_action, action_expert_dim)
        """
        residual = x
        x = self.norm(x)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return residual + x
