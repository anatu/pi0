"""Shared multi-head self-attention with blockwise causal masking."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def build_blockwise_causal_mask(
    n_block1: int,
    n_block2: int,
    n_block3: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build the blockwise causal attention mask for π0.

    Masking rules:
        Block 1 [image + language]: attends only to Block 1 (bidirectional within)
        Block 2 [proprio q_t]:      attends to Block 1 + Block 2
        Block 3 [noisy actions]:    attends to all blocks (bidirectional within Block 3)

    Tokens CANNOT attend to tokens in later blocks.

    Args:
        n_block1: Number of tokens in Block 1 (image + language)
        n_block2: Number of tokens in Block 2 (proprio, typically 1)
        n_block3: Number of tokens in Block 3 (action chunk, typically H)
        device: Torch device
    Returns:
        (N_total, N_total) bool mask where True = ALLOWED to attend.
        N_total = n_block1 + n_block2 + n_block3
    """
    N = n_block1 + n_block2 + n_block3
    # Start with all False (blocked)
    mask = torch.zeros(N, N, dtype=torch.bool, device=device)

    b1_end = n_block1
    b2_end = n_block1 + n_block2
    b3_end = N

    # Block 1 attends to Block 1 (bidirectional)
    mask[:b1_end, :b1_end] = True

    # Block 2 attends to Block 1 + Block 2
    mask[b1_end:b2_end, :b2_end] = True

    # Block 3 attends to Block 1 + Block 2 + Block 3 (bidirectional within Block 3)
    mask[b2_end:b3_end, :b3_end] = True

    return mask


class SharedMultiHeadAttention(nn.Module):
    """Standard multi-head self-attention that accepts an external attention mask.

    All tokens (from both backbone and action expert) are projected into a shared
    attention dimension, attend to each other through shared Q/K/V projections,
    then the output is projected back.

    Since backbone tokens (dim=d) and action expert tokens (dim=w) have different
    widths, the caller must project them into a common dimension before calling
    this module.

    Input:  (B, N_total, attention_dim) — all tokens concatenated
    Output: (B, N_total, attention_dim)
    """

    def __init__(self, attention_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert attention_dim % num_heads == 0, (
            f"attention_dim {attention_dim} not divisible by num_heads {num_heads}"
        )
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(attention_dim, attention_dim)
        self.k_proj = nn.Linear(attention_dim, attention_dim)
        self.v_proj = nn.Linear(attention_dim, attention_dim)
        self.out_proj = nn.Linear(attention_dim, attention_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, attention_dim)
            mask: (N, N) or (B, N, N) bool mask. True = allowed to attend.
        Returns:
            (B, N, attention_dim)
        """
        B, N, D = x.shape

        # Project to Q, K, V and reshape for multi-head
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        # q, k, v: (B, num_heads, N, head_dim)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        # attn_weights: (B, num_heads, N, N)

        if mask is not None:
            # Expand mask for head dimension
            if mask.dim() == 2:
                # (N, N) -> (1, 1, N, N)
                attn_mask = mask.unsqueeze(0).unsqueeze(0)
            else:
                # (B, N, N) -> (B, 1, N, N)
                attn_mask = mask.unsqueeze(1)
            attn_weights = attn_weights.masked_fill(~attn_mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        out = torch.matmul(attn_weights, v)  # (B, num_heads, N, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)
