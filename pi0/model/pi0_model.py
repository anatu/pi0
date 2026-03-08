"""Top-level π0 model assembling all components.

Architecture per transformer layer:
    1. Project all tokens to shared attention_dim (768)
    2. Shared self-attention with blockwise causal mask
    3. Project attention output back to per-expert dims
    4. Route tokens to their respective expert FFNs
    5. Repeat L times
    6. Extract action token outputs → action head → velocity field
"""

import torch
import torch.nn as nn

from pi0.config import ModelConfig, FlowConfig
from pi0.model.token_embed import ImageTokenizer, ProprioEmbedding, LanguageEmbedding
from pi0.model.timestep_embed import ActionTokenEmbedding
from pi0.model.attention import SharedMultiHeadAttention, build_blockwise_causal_mask
from pi0.model.backbone import BackboneFFN
from pi0.model.action_expert import ActionExpertFFN
from pi0.model.action_head import ActionHead


class Pi0TransformerLayer(nn.Module):
    """A single π0 transformer layer: shared attention + dual expert FFNs.

    Handles the dimension mismatch between backbone (d=768) and action expert (w=256):
    - Before attention: project action expert tokens w→d (up-project)
    - Shared attention at dim d
    - After attention: project action expert tokens d→w (down-project)
    - Route to respective FFNs
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        d = config.attention_dim  # = backbone_dim = 768
        w = config.action_expert_dim  # = 256

        # Pre-attention layer norms (one per expert's native dim)
        self.backbone_pre_norm = nn.LayerNorm(d)
        self.action_pre_norm = nn.LayerNorm(w)

        # Projection layers for action expert tokens to/from shared attention dim
        self.action_up_proj = nn.Linear(w, d)
        self.action_down_proj = nn.Linear(d, w)

        # Shared attention (operates at attention_dim = d)
        self.attention = SharedMultiHeadAttention(d, config.num_heads)

        # Expert FFNs
        self.backbone_ffn = BackboneFFN(d, config.backbone_mlp_dim)
        self.action_ffn = ActionExpertFFN(w, config.action_expert_mlp_dim)

    def forward(
        self,
        backbone_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            backbone_tokens: (B, N_backbone, d) — image + language tokens
            action_tokens: (B, N_action, w) — proprio + noisy action tokens
            mask: (N_total, N_total) blockwise causal mask
        Returns:
            (backbone_tokens, action_tokens) with same shapes as input
        """
        # --- Shared Attention ---
        # Normalize in native dims
        bb_normed = self.backbone_pre_norm(backbone_tokens)
        act_normed = self.action_pre_norm(action_tokens)

        # Project action tokens up to attention dim
        act_up = self.action_up_proj(act_normed)

        # Concatenate all tokens at shared attention dim
        all_tokens = torch.cat([bb_normed, act_up], dim=1)

        # Apply shared attention with mask
        attn_out = self.attention(all_tokens, mask=mask)

        # Split back into backbone and action portions
        N_bb = backbone_tokens.shape[1]
        bb_attn_out = attn_out[:, :N_bb, :]
        act_attn_out = attn_out[:, N_bb:, :]

        # Residual connection for backbone tokens (already at dim d)
        backbone_tokens = backbone_tokens + bb_attn_out

        # Project action attention output back down, then residual
        action_tokens = action_tokens + self.action_down_proj(act_attn_out)

        # --- Expert FFNs ---
        backbone_tokens = self.backbone_ffn(backbone_tokens)
        action_tokens = self.action_ffn(action_tokens)

        return backbone_tokens, action_tokens


class Pi0Model(nn.Module):
    """Full π0 Vision-Language-Action model.

    Takes image observations, language commands, proprioceptive state,
    noisy actions, and flow matching timestep. Outputs the predicted
    velocity field for flow matching.
    """

    def __init__(self, model_config: ModelConfig, flow_config: FlowConfig):
        super().__init__()
        self.config = model_config

        # --- Token Embeddings ---
        self.image_tokenizer = ImageTokenizer(
            clip_model_name=model_config.clip_model_name,
            backbone_dim=model_config.backbone_dim,
            image_size=model_config.image_size,
            freeze=model_config.freeze_image_encoder,
        )
        self.language_embedding = LanguageEmbedding(
            vocab_size=49408,  # CLIP tokenizer vocab size
            backbone_dim=model_config.backbone_dim,
        )
        self.proprio_embedding = ProprioEmbedding(
            proprio_dim=model_config.proprio_dim,
            action_expert_dim=model_config.action_expert_dim,
        )
        self.action_token_embedding = ActionTokenEmbedding(
            action_dim=model_config.action_dim,
            action_expert_dim=model_config.action_expert_dim,
            timestep_embed_dim=flow_config.timestep_embed_dim,
        )

        # --- Transformer Layers ---
        self.layers = nn.ModuleList([
            Pi0TransformerLayer(model_config)
            for _ in range(model_config.num_layers)
        ])

        # --- Action Output Head ---
        self.action_head = ActionHead(
            action_expert_dim=model_config.action_expert_dim,
            action_dim=model_config.action_dim,
        )

        # Cache for the attention mask (built once per forward, reused)
        self._cached_mask = None
        self._cached_mask_key = None

    def forward(
        self,
        images: torch.Tensor,
        language_tokens: torch.Tensor,
        proprio: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W) float32 in [0, 1]
            language_tokens: (B, N_lang) int64 token IDs
            proprio: (B, d_proprio) float32
            noisy_actions: (B, H, d_action) float32 noisy action chunk
            timestep: (B,) or scalar, flow matching timestep τ ∈ [0, 1]
        Returns:
            (B, H, d_action) predicted velocity field v_θ
        """
        # --- Embed tokens ---
        # Backbone tokens (image + language) at dim d
        img_tokens = self.image_tokenizer(images)  # (B, N_img, d)
        lang_tokens = self.language_embedding(language_tokens)  # (B, N_lang, d)
        backbone_tokens = torch.cat([img_tokens, lang_tokens], dim=1)  # (B, N_bb, d)

        # Action expert tokens (proprio + noisy actions) at dim w
        proprio_tokens = self.proprio_embedding(proprio)  # (B, 1, w)
        action_tokens = self.action_token_embedding(noisy_actions, timestep)  # (B, H, w)
        action_expert_tokens = torch.cat([proprio_tokens, action_tokens], dim=1)  # (B, 1+H, w)

        # --- Build attention mask ---
        N_bb = backbone_tokens.shape[1]
        N_action = action_expert_tokens.shape[1]
        # Block1 = img+lang, Block2 = proprio (1 token), Block3 = actions (H tokens)
        n_block1 = N_bb
        n_block2 = 1
        n_block3 = self.config.action_chunk_length

        mask_key = (n_block1, n_block2, n_block3)
        if self._cached_mask_key != mask_key or self._cached_mask is None:
            self._cached_mask = build_blockwise_causal_mask(
                n_block1, n_block2, n_block3, device=backbone_tokens.device
            )
            self._cached_mask_key = mask_key
        mask = self._cached_mask

        # --- Transformer layers ---
        for layer in self.layers:
            backbone_tokens, action_expert_tokens = layer(
                backbone_tokens, action_expert_tokens, mask
            )

        # --- Extract action token outputs and decode ---
        # action_expert_tokens: (B, 1+H, w) — skip the proprio token, take action tokens
        action_outputs = action_expert_tokens[:, 1:, :]  # (B, H, w)
        velocity_field = self.action_head(action_outputs)  # (B, H, d_action)

        return velocity_field
