"""CLI script to evaluate the trained π0 model and baselines."""

import argparse

import numpy as np

from pi0.config import EnvConfig, FlowConfig
from pi0.eval.evaluator import Evaluator, Pi0Policy
from pi0.eval.baselines import RandomPolicy, BCMLPBaseline
from pi0.env.expert_policy import ExpertPolicy
from pi0.training.trainer import Trainer
from pi0.data.dataset import TrajectoryDataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate π0 and baselines")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--data_dir", type=str, default="data/trajectories",
                        help="Data dir for training BC-MLP baseline")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--skip_bc", action="store_true", help="Skip BC-MLP baseline")
    parser.add_argument("--bc_epochs", type=int, default=50)
    args = parser.parse_args()

    env_cfg = EnvConfig()
    evaluator = Evaluator(env_cfg)

    results = {}

    # ── Scripted Expert ──
    print("Evaluating: Scripted Expert")
    expert = ExpertPolicy(config=env_cfg)
    results["Expert"] = evaluator.evaluate(expert, num_episodes=args.episodes)

    # ── Random Policy ──
    print("Evaluating: Random Policy")
    random_pol = RandomPolicy(config=env_cfg)
    results["Random"] = evaluator.evaluate(random_pol, num_episodes=args.episodes)

    # ── Trained π0 ──
    print(f"Evaluating: π0 (from {args.checkpoint})")
    model, model_cfg, flow_cfg = Trainer.load_model_from_checkpoint(
        args.checkpoint, device=args.device
    )
    pi0_policy = Pi0Policy(model, model_cfg, flow_cfg, device=args.device)
    results["Pi0"] = evaluator.evaluate(pi0_policy, num_episodes=args.episodes)

    # ── BC-MLP Baseline ──
    if not args.skip_bc:
        print("Training and evaluating: BC-MLP Baseline")
        bc = BCMLPBaseline(device=args.device)
        dataset = TrajectoryDataset(args.data_dir, chunk_length=16)
        bc.train_baseline(dataset, epochs=args.bc_epochs, num_workers=0)
        results["BC-MLP"] = evaluator.evaluate(bc, num_episodes=args.episodes)

    # ── Print Results Table ──
    print("\n" + "=" * 75)
    print(f"{'Policy':<12} {'Success%':>10} {'Avg Reward':>12} {'± Std':>8} "
          f"{'Avg Length':>12} {'± Std':>8}")
    print("-" * 75)
    for name, m in results.items():
        print(
            f"{name:<12} {m['success_rate']*100:>9.1f}% "
            f"{m['avg_reward']:>12.2f} {m['std_reward']:>8.2f} "
            f"{m['avg_episode_length']:>12.1f} {m['std_episode_length']:>8.1f}"
        )
    print("=" * 75)


if __name__ == "__main__":
    main()
