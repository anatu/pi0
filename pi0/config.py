"""Configuration dataclasses for all π0 hyperparameters."""

from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    workspace_min: float = 0.0
    workspace_max: float = 1.0
    image_size: int = 64
    proprio_dim: int = 4  # [x, y, vx, vy]
    action_dim: int = 2  # [dvx, dvy]
    max_episode_steps: int = 100
    reach_threshold: float = 0.05
    reach_bonus: float = 1.0
    max_speed: float = 0.1
    max_action: float = 0.05
    dt: float = 1.0
    language_command: str = "reach the red target"


@dataclass
class ModelConfig:
    # Image encoder
    clip_model_name: str = "openai/clip-vit-base-patch32"
    image_size: int = 64
    image_token_dim: int = 512  # CLIP ViT-B/32 output dim
    freeze_image_encoder: bool = True

    # Language
    language_model_name: str = "openai/clip-vit-base-patch32"  # Use CLIP tokenizer

    # Backbone (Expert 1)
    backbone_dim: int = 768  # d
    backbone_mlp_dim: int = 3072  # 4*d
    num_layers: int = 6
    num_heads: int = 12

    # Action expert (Expert 2)
    action_expert_dim: int = 256  # w
    action_expert_mlp_dim: int = 1024  # 4*w

    # Action
    action_dim: int = 2
    proprio_dim: int = 4
    action_chunk_length: int = 16  # H

    # Attention
    attention_dim: int = 768  # Shared attention dimension


@dataclass
class FlowConfig:
    beta_alpha: float = 1.5
    beta_beta: float = 1.0
    timestep_cutoff: float = 0.999  # s
    euler_steps: int = 10  # K
    timestep_embed_dim: int = 256


@dataclass
class TrainingConfig:
    # Optimizer
    lr: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Schedule
    warmup_steps: int = 500
    max_epochs: int = 200

    # Batching
    batch_size: int = 64
    num_workers: int = 2

    # Logging & checkpointing
    log_every: int = 50  # steps
    eval_every: int = 10  # epochs
    checkpoint_every: int = 20  # epochs
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"

    # Device
    device: str = "auto"


@dataclass
class DataConfig:
    num_trajectories: int = 2000
    output_dir: str = "data/trajectories"
    action_chunk_length: int = 16  # H


@dataclass
class EvalConfig:
    num_episodes: int = 100
    num_vis_episodes: int = 5
    output_dir: str = "outputs"
