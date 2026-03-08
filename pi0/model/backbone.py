"""VLM backbone (Expert 1): FFN layers for image and language tokens."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BackboneFFN(nn.Module):
    """Feed-forward network for Expert 1 (VLM backbone).

    Standard transformer FFN: LayerNorm → Linear → GELU → Linear → residual.
    Processes image and language tokens at backbone_dim width.
    """

    def __init__(self, backbone_dim: int, mlp_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(backbone_dim)
        self.fc1 = nn.Linear(backbone_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, backbone_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N_backbone, backbone_dim)
        Returns:
            (B, N_backbone, backbone_dim)
        """
        residual = x
        x = self.norm(x)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return residual + x
