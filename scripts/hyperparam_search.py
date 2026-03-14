"""Hyperparameter grid search for the π0 model.

Trains multiple configurations, evaluates each, and produces a comparison table.
Results are saved to a JSON file for later analysis.
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import torch

from pi0.config import ModelConfig, FlowConfig, TrainingConfig, EnvConfig
from pi0.training.trainer import Trainer
from pi0.eval.evaluator import Evaluator, Pi0Policy
from pi0.env.expert_policy import ExpertPolicy


GRID = {
    "lr": [5e-5, 1e-4, 3e-4],
    "batch_size": [32, 64],
    "warmup_steps": [200, 500],
}


def build_configs(hp: dict) -> tuple[ModelConfig, FlowConfig, TrainingConfig]:
    """Create config objects from a hyperparameter dict."""
    model_cfg = ModelConfig()
    flow_cfg = FlowConfig()
    train_cfg = TrainingConfig(
        lr=hp["lr"],
        batch_size=hp["batch_size"],
        warmup_steps=hp["warmup_steps"],
        max_epochs=hp["epochs"],
        log_every=hp.get("log_every", 100),
        eval_every=hp["epochs"] + 1,  # disable mid-training eval
        checkpoint_every=hp["epochs"] + 1,  # only save final
        checkpoint_dir=hp["checkpoint_dir"],
        log_dir=hp["log_dir"],
        device=hp["device"],
        num_workers=hp.get("num_workers", 2),
    )
    return model_cfg, flow_cfg, train_cfg


def evaluate_model(model, model_cfg, flow_cfg, device, num_episodes):
    """Evaluate a trained model and return metrics dict."""
    env_cfg = EnvConfig()
    evaluator = Evaluator(env_cfg)
    pi0_policy = Pi0Policy(model, model_cfg, flow_cfg, device=device)
    return evaluator.evaluate(pi0_policy, num_episodes=num_episodes)


def run_search(args):
    grid_keys = sorted(GRID.keys())
    grid_values = [GRID[k] for k in grid_keys]
    combos = list(itertools.product(*grid_values))

    print(f"Hyperparameter grid search: {len(combos)} configurations")
    print(f"  Parameters: {grid_keys}")
    print(f"  Epochs per config: {args.epochs}")
    print(f"  Eval episodes: {args.eval_episodes}")
    print()

    # Get expert baseline once
    env_cfg = EnvConfig()
    evaluator = Evaluator(env_cfg)
    expert = ExpertPolicy(config=env_cfg)
    expert_metrics = evaluator.evaluate(expert, num_episodes=args.eval_episodes)
    print(f"Expert baseline: success={expert_metrics['success_rate']*100:.0f}%, "
          f"avg_reward={expert_metrics['avg_reward']:.2f}")
    print()

    results = []

    for i, combo in enumerate(combos):
        hp = dict(zip(grid_keys, combo))
        hp["epochs"] = args.epochs
        hp["device"] = args.device
        hp["num_workers"] = args.num_workers

        run_name = "_".join(f"{k}={v}" for k, v in sorted(hp.items())
                           if k in grid_keys)
        run_dir = Path(args.output_dir) / run_name
        hp["checkpoint_dir"] = str(run_dir / "checkpoints")
        hp["log_dir"] = str(run_dir / "logs")

        print(f"{'='*70}")
        print(f"Config {i+1}/{len(combos)}: {run_name}")
        print(f"{'='*70}")

        model_cfg, flow_cfg, train_cfg = build_configs(hp)

        # Train
        t0 = time.time()
        trainer = Trainer(
            model_config=model_cfg,
            flow_config=flow_cfg,
            training_config=train_cfg,
            data_dir=args.data_dir,
        )
        trainer.train()
        train_time = time.time() - t0

        # Evaluate
        print(f"Evaluating {run_name} ...")
        trainer.model.eval()
        metrics = evaluate_model(
            trainer.model, model_cfg, flow_cfg,
            device=args.device, num_episodes=args.eval_episodes,
        )

        result = {
            "config": {k: hp[k] for k in grid_keys},
            "run_name": run_name,
            "train_time_s": round(train_time, 1),
            "success_rate": metrics["success_rate"],
            "avg_reward": metrics["avg_reward"],
            "std_reward": metrics["std_reward"],
            "avg_episode_length": metrics["avg_episode_length"],
        }
        results.append(result)

        print(f"  -> success={metrics['success_rate']*100:.0f}%, "
              f"avg_reward={metrics['avg_reward']:.2f}, "
              f"time={train_time:.0f}s")
        print()

        # Free GPU memory between runs
        del trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save results
    output_path = Path(args.output_dir) / "results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_results = {
        "expert": {
            "success_rate": expert_metrics["success_rate"],
            "avg_reward": expert_metrics["avg_reward"],
            "std_reward": expert_metrics["std_reward"],
            "avg_episode_length": expert_metrics["avg_episode_length"],
        },
        "grid": GRID,
        "epochs": args.epochs,
        "eval_episodes": args.eval_episodes,
        "runs": results,
    }
    with open(output_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"Results saved to {output_path}")

    # Print comparison table
    print()
    print_results_table(full_results)

    # Best config
    best = max(results, key=lambda r: r["avg_reward"])
    print(f"\nBest config: {best['run_name']}")
    print(f"  avg_reward={best['avg_reward']:.2f}, "
          f"success={best['success_rate']*100:.0f}%")


def print_results_table(data):
    """Print a formatted comparison table."""
    runs = data["runs"]
    expert = data["expert"]

    print("=" * 90)
    print(f"{'Config':<35} {'Success%':>9} {'Avg Reward':>11} {'± Std':>7} "
          f"{'Avg Len':>8} {'Time(s)':>8}")
    print("-" * 90)
    print(f"{'Expert (oracle)':<35} {expert['success_rate']*100:>8.1f}% "
          f"{expert['avg_reward']:>11.2f} {expert['std_reward']:>7.2f} "
          f"{expert['avg_episode_length']:>8.1f} {'—':>8}")

    # Sort by avg_reward descending (less negative = better)
    for r in sorted(runs, key=lambda x: x["avg_reward"], reverse=True):
        cfg = r["config"]
        label = f"lr={cfg['lr']},bs={cfg['batch_size']},wu={cfg['warmup_steps']}"
        print(f"{label:<35} {r['success_rate']*100:>8.1f}% "
              f"{r['avg_reward']:>11.2f} {r['std_reward']:>7.2f} "
              f"{r['avg_episode_length']:>8.1f} {r['train_time_s']:>8.1f}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="π0 hyperparameter grid search")
    parser.add_argument("--data_dir", type=str, default="data/trajectories")
    parser.add_argument("--output_dir", type=str, default="hp_search")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs per config (default: 50)")
    parser.add_argument("--eval_episodes", type=int, default=10,
                        help="Evaluation episodes per config (default: 10)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--results_only", type=str, default=None,
                        help="Path to results.json to just print the table")
    args = parser.parse_args()

    if args.results_only:
        with open(args.results_only) as f:
            data = json.load(f)
        print_results_table(data)
        return

    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    run_search(args)


if __name__ == "__main__":
    main()
