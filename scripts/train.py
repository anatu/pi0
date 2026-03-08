"""CLI script to train the π0 model."""

import argparse

from pi0.config import ModelConfig, FlowConfig, TrainingConfig, EnvConfig
from pi0.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train π0 model")
    parser.add_argument("--data_dir", type=str, default="data/trajectories")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--checkpoint_every", type=int, default=20)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    model_cfg = ModelConfig()
    flow_cfg = FlowConfig()
    train_cfg = TrainingConfig(
        lr=args.lr,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        log_every=args.log_every,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        device=args.device,
        num_workers=args.num_workers,
    )

    trainer = Trainer(
        model_config=model_cfg,
        flow_config=flow_cfg,
        training_config=train_cfg,
        data_dir=args.data_dir,
    )
    trainer.train()


if __name__ == "__main__":
    main()
