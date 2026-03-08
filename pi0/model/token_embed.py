"""Token embedding modules: image, proprioception, and language projections."""

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPProcessor


class ImageTokenizer(nn.Module):
    """Encode images with a frozen CLIP ViT and project to backbone dim.

    Input:  (B, 3, H, W) float32 images in [0, 1]
    Output: (B, N_img, backbone_dim) token embeddings

    N_img = (image_size / patch_size)^2 + 1  (patch tokens + CLS).
    For CLIP ViT-B/32 with 64×64 input: (64/32)^2 + 1 = 5 tokens.
    We drop the CLS token and keep only the 4 patch tokens.
    """

    def __init__(
        self,
        clip_model_name: str,
        backbone_dim: int,
        image_size: int = 64,
        freeze: bool = True,
    ):
        super().__init__()
        self.vit = CLIPVisionModel.from_pretrained(clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)

        # Determine the hidden size from the loaded model
        vit_hidden = self.vit.config.hidden_size  # 768 for ViT-B/32

        self.proj = nn.Linear(vit_hidden, backbone_dim)

        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W) float32 in [0, 1]
        Returns:
            (B, N_img, backbone_dim) image token embeddings
        """
        # CLIP ViT expects pixel_values in its own normalized range,
        # but we pass raw [0,1] tensors — the model still works,
        # just without the exact ImageNet normalization. Fine for our use.
        with torch.no_grad():
            outputs = self.vit(pixel_values=images, interpolate_pos_encoding=True)
        # last_hidden_state: (B, N_patches+1, hidden_size)
        # Drop CLS token (index 0), keep patch tokens
        patch_tokens = outputs.last_hidden_state[:, 1:, :]
        return self.proj(patch_tokens)


class ProprioEmbedding(nn.Module):
    """Project proprioceptive state vector into action expert embedding space.

    Input:  (B, d_proprio)
    Output: (B, 1, action_expert_dim)  — a single token
    """

    def __init__(self, proprio_dim: int, action_expert_dim: int):
        super().__init__()
        self.proj = nn.Linear(proprio_dim, action_expert_dim)

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            proprio: (B, d_proprio)
        Returns:
            (B, 1, action_expert_dim)
        """
        return self.proj(proprio).unsqueeze(1)


class LanguageEmbedding(nn.Module):
    """Embed language token IDs into backbone dimension.

    Uses a learned embedding table. Token IDs come from the CLIP tokenizer.

    Input:  (B, N_lang) int64 token IDs
    Output: (B, N_lang, backbone_dim)
    """

    def __init__(self, vocab_size: int, backbone_dim: int, max_seq_len: int = 77):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, backbone_dim)
        self.position_embed = nn.Embedding(max_seq_len, backbone_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (B, N_lang) int64
        Returns:
            (B, N_lang, backbone_dim)
        """
        B, N = token_ids.shape
        positions = torch.arange(N, device=token_ids.device).unsqueeze(0).expand(B, -1)
        return self.token_embed(token_ids) + self.position_embed(positions)
