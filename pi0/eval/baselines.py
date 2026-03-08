"""Baseline policies: random and behavioral cloning MLP."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pi0.config import EnvConfig


class RandomPolicy:
    """Uniform random actions within the action space bounds."""

    def __init__(self, config: EnvConfig | None = None):
        self.cfg = config or EnvConfig()
        self.rng = np.random.default_rng(0)

    def act(self, obs: dict) -> np.ndarray:
        return self.rng.uniform(
            -self.cfg.max_action,
            self.cfg.max_action,
            size=self.cfg.action_dim,
        ).astype(np.float32)


class BCMLPBaseline:
    """Simple 3-layer MLP behavioral cloning baseline.

    Predicts a single next action from flattened [image_features, proprio].
    Uses a frozen CLIP ViT to extract image features, then feeds
    concat(image_features, proprio) into a 3-layer MLP.

    This baseline isolates the value of the π0 architecture by using
    the same data but no flow matching and no action expert.
    """

    def __init__(
        self,
        image_feature_dim: int = 768,  # CLIP ViT-B/32 hidden_size
        proprio_dim: int = 4,
        action_dim: int = 2,
        hidden_dim: int = 256,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.image_feature_dim = image_feature_dim
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        input_dim = image_feature_dim + proprio_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        ).to(self.device)

        # Image encoder (frozen CLIP)
        self._image_encoder = None
        self._mlp_initialized = False

    def _get_image_encoder(self):
        """Lazy-load the CLIP image encoder."""
        if self._image_encoder is None:
            from transformers import CLIPVisionModel
            self._image_encoder = CLIPVisionModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            ).to(self.device)
            self._image_encoder.eval()
            for p in self._image_encoder.parameters():
                p.requires_grad = False
        return self._image_encoder

    @torch.no_grad()
    def _encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Extract image features using CLIP.

        Args:
            image: (B, 3, H, W) float32 in [0, 1]
        Returns:
            (B, image_feature_dim) pooled features.
        """
        encoder = self._get_image_encoder()
        outputs = encoder(pixel_values=image, interpolate_pos_encoding=True)
        # Use the CLS token output as the pooled feature
        return outputs.last_hidden_state[:, 0, :]  # (B, 768)

    def train_baseline(
        self,
        dataset,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        num_workers: int = 0,
    ):
        """Train the BC-MLP on the same trajectory dataset.

        Only uses the first action from each action chunk (single-step prediction).
        """
        from pi0.training.trainer import _collate_fn

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=_collate_fn,
            drop_last=True,
        )

        # Determine image feature dim from first batch, re-init MLP if needed
        sample_batch = next(iter(loader))
        sample_img = sample_batch["image"][:1].to(self.device)
        img_feat = self._encode_image(sample_img)
        self._ensure_mlp_matches_encoder(img_feat)

        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=lr)
        self.mlp.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch in loader:
                images = batch["image"].to(self.device)
                proprio = batch["proprio"].to(self.device)
                actions = batch["action_chunk"][:, 0, :].to(self.device)  # first action only

                img_features = self._encode_image(images)
                x = torch.cat([img_features, proprio], dim=-1)
                pred = self.mlp(x)

                loss = nn.functional.mse_loss(pred, actions)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 10 == 0:
                print(f"  BC-MLP epoch {epoch+1}/{epochs} | loss {epoch_loss/n_batches:.4f}")

    def _ensure_mlp_matches_encoder(self, img_features: torch.Tensor):
        """Re-initialize MLP if image feature dim doesn't match."""
        actual_dim = img_features.shape[1]
        if actual_dim != self.image_feature_dim or not self._mlp_initialized:
            self.image_feature_dim = actual_dim
            input_dim = actual_dim + self.proprio_dim
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.action_dim),
            ).to(self.device)
            self._mlp_initialized = True

    def act(self, obs: dict) -> np.ndarray:
        """Predict action from observation."""
        self.mlp.eval()

        image = torch.from_numpy(obs["image"]).float() / 255.0
        image = image.permute(2, 0, 1).unsqueeze(0).to(self.device)
        proprio = torch.from_numpy(obs["proprio"]).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            img_features = self._encode_image(image)
            self._ensure_mlp_matches_encoder(img_features)
            x = torch.cat([img_features, proprio], dim=-1)
            action = self.mlp(x)

        return action[0].cpu().numpy()
