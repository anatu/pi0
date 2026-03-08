"""Flow matching loss with shifted beta timestep sampling."""

import torch
import torch.nn as nn

from pi0.config import FlowConfig


def sample_timestep(batch_size: int, config: FlowConfig, device: torch.device | None = None) -> torch.Tensor:
    """Sample timesteps from a shifted Beta distribution.

    p(τ) = Beta((s - τ)/s; α, β)  where s = 0.999
    This emphasizes lower (noisier) timesteps.

    We sample u ~ Beta(α, β) and then compute τ = s * (1 - u)
    so that higher beta samples (near 1) map to lower τ (noisier).

    Actually, re-reading the spec: p(τ) = Beta((s-τ)/s; 1.5, 1)
    Let v = (s - τ)/s ~ Beta(1.5, 1), so τ = s*(1 - v) = s - s*v.
    Since v ~ Beta(1.5,1), pdf(v) ∝ v^0.5, so v is biased toward 1,
    meaning τ is biased toward 0 (noisier). This is correct.

    Args:
        batch_size: Number of timesteps to sample.
        config: FlowConfig with beta_alpha, beta_beta, timestep_cutoff.
        device: Torch device.
    Returns:
        (batch_size,) tensor of τ values in [0, s].
    """
    s = config.timestep_cutoff
    # v ~ Beta(alpha, beta), v in [0, 1]
    dist = torch.distributions.Beta(config.beta_alpha, config.beta_beta)
    v = dist.sample((batch_size,))
    # τ = s * (1 - v), so when v~1 → τ~0 (noisy), when v~0 → τ~s (clean)
    tau = s * (1.0 - v)
    if device is not None:
        tau = tau.to(device)
    return tau


def interpolate_actions(
    clean_actions: torch.Tensor,
    noise: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    """Compute noisy actions via linear interpolation.

    A_τ = τ * A + (1 - τ) * ε

    Args:
        clean_actions: (B, H, d_action) ground truth action chunks.
        noise: (B, H, d_action) Gaussian noise ε ~ N(0, I).
        tau: (B,) timestep values.
    Returns:
        (B, H, d_action) noisy actions.
    """
    # Reshape tau for broadcasting: (B,) → (B, 1, 1)
    tau_expanded = tau.view(-1, 1, 1)
    return tau_expanded * clean_actions + (1.0 - tau_expanded) * noise


def compute_target_velocity(
    clean_actions: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Compute the target velocity field.

    u(A_τ | A) = A - ε

    Args:
        clean_actions: (B, H, d_action) ground truth action chunks.
        noise: (B, H, d_action) Gaussian noise.
    Returns:
        (B, H, d_action) target velocity field.
    """
    return clean_actions - noise


class FlowMatchingLoss(nn.Module):
    """Compute the conditional flow matching loss.

    L(θ) = E ||v_θ(A_τ, o) - u(A_τ|A)||²

    where:
        A_τ = τ*A + (1-τ)*ε  (noisy actions)
        u = A - ε             (target velocity)
        v_θ = model output    (predicted velocity)
    """

    def __init__(self, config: FlowConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        model: nn.Module,
        images: torch.Tensor,
        language_tokens: torch.Tensor,
        proprio: torch.Tensor,
        clean_actions: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the flow matching loss for a batch.

        Args:
            model: Pi0Model that takes (images, language_tokens, proprio, noisy_actions, tau)
                   and returns predicted velocity (B, H, d_action).
            images: (B, 3, H_img, W_img)
            language_tokens: (B, N_lang)
            proprio: (B, d_proprio)
            clean_actions: (B, H, d_action) ground truth action chunks.
            padding_mask: (B, H) bool, True = valid. If provided, loss is masked.
        Returns:
            Scalar loss tensor.
        """
        B = clean_actions.shape[0]
        device = clean_actions.device

        # Sample timesteps
        tau = sample_timestep(B, self.config, device=device)

        # Sample noise
        noise = torch.randn_like(clean_actions)

        # Compute noisy actions and target velocity
        noisy_actions = interpolate_actions(clean_actions, noise, tau)
        target_velocity = compute_target_velocity(clean_actions, noise)

        # Forward pass through model
        predicted_velocity = model(images, language_tokens, proprio, noisy_actions, tau)

        # MSE loss
        error = (predicted_velocity - target_velocity) ** 2  # (B, H, d_action)

        if padding_mask is not None:
            # Expand mask: (B, H) → (B, H, 1)
            mask = padding_mask.unsqueeze(-1).float()
            error = error * mask
            # Mean over valid entries only
            loss = error.sum() / (mask.sum() * clean_actions.shape[-1] + 1e-8)
        else:
            loss = error.mean()

        return loss
