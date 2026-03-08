"""Action output head: project action expert outputs to action space."""

import torch
import torch.nn as nn


class ActionHead(nn.Module):
    """Linear projection from action expert output to predicted velocity field.

    Input:  (B, H, action_expert_dim) — transformer outputs for action tokens
    Output: (B, H, action_dim) — predicted velocity field v_θ
    """

    def __init__(self, action_expert_dim: int, action_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(action_expert_dim)
        self.proj = nn.Linear(action_expert_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))
