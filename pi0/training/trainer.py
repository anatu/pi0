"""Training loop for the π0 model."""

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import CLIPTokenizer

from pi0.config import ModelConfig, FlowConfig, TrainingConfig, EnvConfig
from pi0.model.pi0_model import Pi0Model
from pi0.flow.flow_matching import FlowMatchingLoss
from pi0.flow.sampler import EulerSampler
from pi0.data.dataset import TrajectoryDataset
from pi0.training.scheduler import get_cosine_schedule_with_warmup


def _collate_fn(batch: list[dict]) -> dict:
    """Custom collate that handles the language string field.

    Stacks tensors normally but collects language strings into a list.
    """
    collated = {}
    collated["image"] = torch.stack([b["image"] for b in batch])
    collated["proprio"] = torch.stack([b["proprio"] for b in batch])
    collated["action_chunk"] = torch.stack([b["action_chunk"] for b in batch])
    collated["padding_mask"] = torch.stack([b["padding_mask"] for b in batch])
    collated["language"] = [b["language"] for b in batch]
    return collated


class Trainer:
    """Full training pipeline for π0.

    Handles: data loading, tokenization, training loop, logging,
    checkpointing, and periodic evaluation.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        flow_config: FlowConfig,
        training_config: TrainingConfig,
        data_dir: str,
        env_config: EnvConfig | None = None,
    ):
        self.model_cfg = model_config
        self.flow_cfg = flow_config
        self.train_cfg = training_config
        self.env_cfg = env_config or EnvConfig()

        # Device
        if training_config.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(training_config.device)

        # Model
        self.model = Pi0Model(model_config, flow_config).to(self.device)

        # Loss
        self.loss_fn = FlowMatchingLoss(flow_config)

        # Tokenizer for language commands
        self.tokenizer = CLIPTokenizer.from_pretrained(model_config.clip_model_name)

        # Dataset and dataloader
        self.dataset = TrajectoryDataset(
            data_dir=data_dir,
            chunk_length=model_config.action_chunk_length,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=training_config.batch_size,
            shuffle=True,
            num_workers=training_config.num_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=_collate_fn,
            drop_last=True,
            persistent_workers=training_config.num_workers > 0,
        )

        # Optimizer (only trainable params)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=training_config.lr,
            weight_decay=training_config.weight_decay,
        )

        # Scheduler
        steps_per_epoch = len(self.dataloader)
        total_steps = steps_per_epoch * training_config.max_epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            warmup_steps=training_config.warmup_steps,
            total_steps=total_steps,
        )

        # Logging
        self.log_dir = Path(training_config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Checkpointing
        self.checkpoint_dir = Path(training_config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0

    def _tokenize_language(self, language_strings: list[str]) -> torch.Tensor:
        """Tokenize a batch of language strings into padded token IDs."""
        encoded = self.tokenizer(
            language_strings,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        return encoded["input_ids"].to(self.device)

    def train_step(self, batch: dict) -> float:
        """Execute a single training step.

        Args:
            batch: Dict from dataloader with keys:
                image, language, proprio, action_chunk, padding_mask
        Returns:
            Loss value as float.
        """
        self.model.train()

        images = batch["image"].to(self.device)
        proprio = batch["proprio"].to(self.device)
        action_chunk = batch["action_chunk"].to(self.device)
        padding_mask = batch["padding_mask"].to(self.device)
        language_tokens = self._tokenize_language(batch["language"])

        # Forward + loss
        loss = self.loss_fn(
            self.model, images, language_tokens, proprio, action_chunk, padding_mask
        )

        # Backward
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.train_cfg.max_grad_norm,
        )

        self.optimizer.step()
        self.scheduler.step()

        self.global_step += 1
        return loss.item()

    def train(self, eval_callback=None):
        """Run the full training loop.

        Args:
            eval_callback: Optional callable(model, epoch) for periodic evaluation.
                           Should return a metrics dict or None.
        """
        print(f"Training on {self.device} | {len(self.dataset)} samples | "
              f"{len(self.dataloader)} batches/epoch")
        print(f"Trainable params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        for epoch in range(1, self.train_cfg.max_epochs + 1):
            epoch_losses = []
            epoch_start = time.time()

            for batch in self.dataloader:
                loss = self.train_step(batch)
                epoch_losses.append(loss)

                # Logging
                if self.global_step % self.train_cfg.log_every == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    self.writer.add_scalar("train/loss", loss, self.global_step)
                    self.writer.add_scalar("train/lr", lr, self.global_step)
                    print(
                        f"  step {self.global_step:>6d} | "
                        f"loss {loss:.4f} | lr {lr:.2e}"
                    )

            # Epoch summary
            epoch_time = time.time() - epoch_start
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(
                f"Epoch {epoch:>3d}/{self.train_cfg.max_epochs} | "
                f"avg_loss {avg_loss:.4f} | "
                f"time {epoch_time:.1f}s"
            )
            self.writer.add_scalar("train/epoch_loss", avg_loss, epoch)

            # Checkpointing
            if epoch % self.train_cfg.checkpoint_every == 0:
                self.save_checkpoint(epoch)

            # Evaluation
            if eval_callback is not None and epoch % self.train_cfg.eval_every == 0:
                metrics = eval_callback(self.model, epoch)
                if metrics:
                    for k, v in metrics.items():
                        self.writer.add_scalar(f"eval/{k}", v, epoch)

        # Save final checkpoint
        self.save_checkpoint("final")
        self.writer.close()
        print("Training complete.")

    def save_checkpoint(self, tag):
        """Save model checkpoint."""
        path = self.checkpoint_dir / f"checkpoint_{tag}.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "model_config": self.model_cfg,
                "flow_config": self.flow_cfg,
            },
            path,
        )
        # Also save as latest
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "model_config": self.model_cfg,
                "flow_config": self.flow_cfg,
            },
            latest_path,
        )
        print(f"  Saved checkpoint: {path}")

    @staticmethod
    def load_model_from_checkpoint(
        checkpoint_path: str,
        device: str = "auto",
    ) -> tuple[Pi0Model, ModelConfig, FlowConfig]:
        """Load a trained model from a checkpoint file.

        Returns:
            (model, model_config, flow_config) tuple.
        """
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_cfg = checkpoint["model_config"]
        flow_cfg = checkpoint["flow_config"]
        model = Pi0Model(model_cfg, flow_cfg).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model, model_cfg, flow_cfg
