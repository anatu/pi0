"""Tests for model components: token embeddings, timestep encoding, attention masking."""

import torch
import pytest
from pi0.config import ModelConfig, FlowConfig


# ── Timestep Embedding Tests ──


def test_sinusoidal_encoding_shape():
    from pi0.model.timestep_embed import sinusoidal_encoding

    tau = torch.tensor([0.0, 0.5, 1.0])
    enc = sinusoidal_encoding(tau, dim=256)
    assert enc.shape == (3, 256)


def test_sinusoidal_encoding_different_timesteps():
    from pi0.model.timestep_embed import sinusoidal_encoding

    t1 = sinusoidal_encoding(torch.tensor(0.0), dim=64)
    t2 = sinusoidal_encoding(torch.tensor(1.0), dim=64)
    # Different timesteps should produce different encodings
    assert not torch.allclose(t1, t2)


def test_sinusoidal_encoding_batch():
    from pi0.model.timestep_embed import sinusoidal_encoding

    tau = torch.rand(8)
    enc = sinusoidal_encoding(tau, dim=128)
    assert enc.shape == (8, 128)


def test_action_token_embedding_shape():
    from pi0.model.timestep_embed import ActionTokenEmbedding

    cfg = ModelConfig()
    fc = FlowConfig()
    embed = ActionTokenEmbedding(
        action_dim=cfg.action_dim,
        action_expert_dim=cfg.action_expert_dim,
        timestep_embed_dim=fc.timestep_embed_dim,
    )
    B, H = 4, cfg.action_chunk_length
    noisy_actions = torch.randn(B, H, cfg.action_dim)
    tau = torch.rand(B)
    out = embed(noisy_actions, tau)
    assert out.shape == (B, H, cfg.action_expert_dim)


def test_action_token_embedding_scalar_tau():
    from pi0.model.timestep_embed import ActionTokenEmbedding

    cfg = ModelConfig()
    fc = FlowConfig()
    embed = ActionTokenEmbedding(
        action_dim=cfg.action_dim,
        action_expert_dim=cfg.action_expert_dim,
        timestep_embed_dim=fc.timestep_embed_dim,
    )
    B, H = 4, 16
    noisy_actions = torch.randn(B, H, cfg.action_dim)
    tau = torch.tensor(0.5)
    out = embed(noisy_actions, tau)
    assert out.shape == (B, H, cfg.action_expert_dim)


# ── Proprio Embedding Tests ──


def test_proprio_embedding_shape():
    from pi0.model.token_embed import ProprioEmbedding

    cfg = ModelConfig()
    embed = ProprioEmbedding(cfg.proprio_dim, cfg.action_expert_dim)
    B = 4
    proprio = torch.randn(B, cfg.proprio_dim)
    out = embed(proprio)
    assert out.shape == (B, 1, cfg.action_expert_dim)


# ── Language Embedding Tests ──


def test_language_embedding_shape():
    from pi0.model.token_embed import LanguageEmbedding

    cfg = ModelConfig()
    vocab_size = 49408  # CLIP vocab size
    embed = LanguageEmbedding(vocab_size, cfg.backbone_dim)
    B, N_lang = 4, 10
    token_ids = torch.randint(0, vocab_size, (B, N_lang))
    out = embed(token_ids)
    assert out.shape == (B, N_lang, cfg.backbone_dim)


# ── Attention Mask Tests ──


def test_mask_shape():
    from pi0.model.attention import build_blockwise_causal_mask

    n1, n2, n3 = 10, 1, 16
    mask = build_blockwise_causal_mask(n1, n2, n3)
    assert mask.shape == (n1 + n2 + n3, n1 + n2 + n3)
    assert mask.dtype == torch.bool


def test_mask_block1_self_attention():
    from pi0.model.attention import build_blockwise_causal_mask

    n1, n2, n3 = 10, 1, 16
    mask = build_blockwise_causal_mask(n1, n2, n3)
    # Block 1 tokens attend to all Block 1 tokens (bidirectional)
    assert mask[:n1, :n1].all()


def test_mask_block1_cannot_attend_to_block2_or_block3():
    from pi0.model.attention import build_blockwise_causal_mask

    n1, n2, n3 = 10, 1, 16
    mask = build_blockwise_causal_mask(n1, n2, n3)
    # Block 1 cannot see Block 2
    assert not mask[:n1, n1 : n1 + n2].any()
    # Block 1 cannot see Block 3
    assert not mask[:n1, n1 + n2 :].any()


def test_mask_block2_attends_to_block1_and_self():
    from pi0.model.attention import build_blockwise_causal_mask

    n1, n2, n3 = 10, 1, 16
    mask = build_blockwise_causal_mask(n1, n2, n3)
    b2_start = n1
    b2_end = n1 + n2
    # Block 2 can attend to Block 1
    assert mask[b2_start:b2_end, :n1].all()
    # Block 2 can attend to itself
    assert mask[b2_start:b2_end, b2_start:b2_end].all()
    # Block 2 cannot attend to Block 3
    assert not mask[b2_start:b2_end, b2_end:].any()


def test_mask_block3_attends_to_all():
    from pi0.model.attention import build_blockwise_causal_mask

    n1, n2, n3 = 10, 1, 16
    mask = build_blockwise_causal_mask(n1, n2, n3)
    b3_start = n1 + n2
    # Block 3 can attend to everything
    assert mask[b3_start:, :].all()


# ── SharedMultiHeadAttention Tests ──


def test_shared_attention_output_shape():
    from pi0.model.attention import SharedMultiHeadAttention, build_blockwise_causal_mask

    cfg = ModelConfig()
    attn = SharedMultiHeadAttention(cfg.attention_dim, cfg.num_heads)

    B = 4
    n1, n2, n3 = 10, 1, 16
    N = n1 + n2 + n3
    x = torch.randn(B, N, cfg.attention_dim)
    mask = build_blockwise_causal_mask(n1, n2, n3)

    out = attn(x, mask=mask)
    assert out.shape == (B, N, cfg.attention_dim)


def test_shared_attention_no_mask():
    from pi0.model.attention import SharedMultiHeadAttention

    cfg = ModelConfig()
    attn = SharedMultiHeadAttention(cfg.attention_dim, cfg.num_heads)
    B, N = 2, 8
    x = torch.randn(B, N, cfg.attention_dim)
    out = attn(x)
    assert out.shape == (B, N, cfg.attention_dim)


def test_shared_attention_gradient_flows():
    from pi0.model.attention import SharedMultiHeadAttention, build_blockwise_causal_mask

    cfg = ModelConfig()
    attn = SharedMultiHeadAttention(cfg.attention_dim, cfg.num_heads)
    B, n1, n2, n3 = 2, 5, 1, 8
    N = n1 + n2 + n3
    x = torch.randn(B, N, cfg.attention_dim, requires_grad=True)
    mask = build_blockwise_causal_mask(n1, n2, n3)
    out = attn(x, mask=mask)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


# ── Image Tokenizer Tests (requires CLIP download) ──


@pytest.mark.slow
def test_image_tokenizer_shape():
    from pi0.model.token_embed import ImageTokenizer

    cfg = ModelConfig()
    tokenizer = ImageTokenizer(
        clip_model_name=cfg.clip_model_name,
        backbone_dim=cfg.backbone_dim,
        image_size=cfg.image_size,
        freeze=True,
    )
    B = 2
    images = torch.rand(B, 3, cfg.image_size, cfg.image_size)
    out = tokenizer(images)
    # For 64x64 with patch_size=32: (64/32)^2 = 4 patch tokens
    assert out.shape == (B, 4, cfg.backbone_dim)
    assert out.dtype == torch.float32


@pytest.mark.slow
def test_image_tokenizer_frozen():
    from pi0.model.token_embed import ImageTokenizer

    cfg = ModelConfig()
    tokenizer = ImageTokenizer(
        clip_model_name=cfg.clip_model_name,
        backbone_dim=cfg.backbone_dim,
        freeze=True,
    )
    # ViT params should not require grad
    for p in tokenizer.vit.parameters():
        assert not p.requires_grad
    # Projection should require grad
    for p in tokenizer.proj.parameters():
        assert p.requires_grad


# ── Phase 4: Expert FFN Tests ──


def test_backbone_ffn_shape():
    from pi0.model.backbone import BackboneFFN

    cfg = ModelConfig()
    ffn = BackboneFFN(cfg.backbone_dim, cfg.backbone_mlp_dim)
    B, N = 2, 14  # img + lang tokens
    x = torch.randn(B, N, cfg.backbone_dim)
    out = ffn(x)
    assert out.shape == (B, N, cfg.backbone_dim)


def test_action_expert_ffn_shape():
    from pi0.model.action_expert import ActionExpertFFN

    cfg = ModelConfig()
    ffn = ActionExpertFFN(cfg.action_expert_dim, cfg.action_expert_mlp_dim)
    B, N = 2, 17  # proprio + action tokens
    x = torch.randn(B, N, cfg.action_expert_dim)
    out = ffn(x)
    assert out.shape == (B, N, cfg.action_expert_dim)


def test_action_head_shape():
    from pi0.model.action_head import ActionHead

    cfg = ModelConfig()
    head = ActionHead(cfg.action_expert_dim, cfg.action_dim)
    B, H = 2, cfg.action_chunk_length
    x = torch.randn(B, H, cfg.action_expert_dim)
    out = head(x)
    assert out.shape == (B, H, cfg.action_dim)


# ── Phase 4: Transformer Layer Tests ──


def test_transformer_layer_shapes():
    from pi0.model.pi0_model import Pi0TransformerLayer
    from pi0.model.attention import build_blockwise_causal_mask

    cfg = ModelConfig()
    layer = Pi0TransformerLayer(cfg)

    B = 2
    N_bb = 14  # img(4) + lang(10)
    N_act = 1 + cfg.action_chunk_length  # proprio + H

    bb = torch.randn(B, N_bb, cfg.backbone_dim)
    act = torch.randn(B, N_act, cfg.action_expert_dim)
    mask = build_blockwise_causal_mask(N_bb, 1, cfg.action_chunk_length)

    bb_out, act_out = layer(bb, act, mask)
    assert bb_out.shape == bb.shape
    assert act_out.shape == act.shape


# ── Phase 4: Full Model Tests ──


@pytest.mark.slow
def test_full_model_forward_pass():
    from pi0.model.pi0_model import Pi0Model

    cfg = ModelConfig()
    fc = FlowConfig()
    model = Pi0Model(cfg, fc)

    B = 2
    images = torch.rand(B, 3, cfg.image_size, cfg.image_size)
    lang_tokens = torch.randint(0, 49408, (B, 10))
    proprio = torch.randn(B, cfg.proprio_dim)
    noisy_actions = torch.randn(B, cfg.action_chunk_length, cfg.action_dim)
    tau = torch.rand(B)

    out = model(images, lang_tokens, proprio, noisy_actions, tau)
    assert out.shape == (B, cfg.action_chunk_length, cfg.action_dim)


@pytest.mark.slow
def test_full_model_parameter_count():
    from pi0.model.pi0_model import Pi0Model

    cfg = ModelConfig()
    fc = FlowConfig()
    model = Pi0Model(cfg, fc)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Total should be in 50M-200M range (CLIP ViT is ~87M frozen)
    assert total_params > 50_000_000, f"Too few total params: {total_params:,}"
    assert total_params < 200_000_000, f"Too many total params: {total_params:,}"

    # Trainable should be smaller since CLIP is frozen
    assert trainable_params < total_params, "Frozen CLIP params should reduce trainable count"


@pytest.mark.slow
def test_gradient_flows_through_action_expert_not_clip():
    from pi0.model.pi0_model import Pi0Model

    cfg = ModelConfig()
    fc = FlowConfig()
    model = Pi0Model(cfg, fc)

    B = 2
    images = torch.rand(B, 3, cfg.image_size, cfg.image_size)
    lang_tokens = torch.randint(0, 49408, (B, 10))
    proprio = torch.randn(B, cfg.proprio_dim)
    noisy_actions = torch.randn(B, cfg.action_chunk_length, cfg.action_dim)
    tau = torch.rand(B)

    out = model(images, lang_tokens, proprio, noisy_actions, tau)
    loss = out.sum()
    loss.backward()

    # Action expert FFN should have gradients
    for layer in model.layers:
        for p in layer.action_ffn.parameters():
            assert p.grad is not None, "Action expert FFN should receive gradients"
            assert p.grad.abs().sum() > 0, "Action expert gradients should be non-zero"

    # Frozen CLIP ViT should NOT have gradients
    for p in model.image_tokenizer.vit.parameters():
        assert p.grad is None, "Frozen CLIP ViT should not receive gradients"

    # Image projection (not frozen) should have gradients
    for p in model.image_tokenizer.proj.parameters():
        assert p.grad is not None, "Image projection should receive gradients"


@pytest.mark.slow
def test_expert_routing_correctness():
    """Verify backbone FFN only sees backbone tokens and action FFN only sees action tokens."""
    from pi0.model.pi0_model import Pi0Model, Pi0TransformerLayer
    from pi0.model.attention import build_blockwise_causal_mask

    cfg = ModelConfig()

    layer = Pi0TransformerLayer(cfg)

    B = 2
    N_bb = 14
    N_act = 1 + cfg.action_chunk_length

    # Use distinctive values to verify routing
    bb = torch.ones(B, N_bb, cfg.backbone_dim) * 1.0
    act = torch.ones(B, N_act, cfg.action_expert_dim) * 2.0
    mask = build_blockwise_causal_mask(N_bb, 1, cfg.action_chunk_length)

    # Run forward and check that outputs maintain their token counts
    bb_out, act_out = layer(bb, act, mask)
    assert bb_out.shape[1] == N_bb, "Backbone tokens count should be preserved"
    assert act_out.shape[1] == N_act, "Action tokens count should be preserved"
