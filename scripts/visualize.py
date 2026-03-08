"""CLI script to generate rollout GIFs for expert, π0, and random policies."""

import argparse
from pathlib import Path

from pi0.config import EnvConfig, FlowConfig
from pi0.eval.visualize import record_and_save_episodes
from pi0.eval.evaluator import Pi0Policy
from pi0.eval.baselines import RandomPolicy
from pi0.env.expert_policy import ExpertPolicy
from pi0.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Generate rollout GIFs")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--scale", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env_cfg = EnvConfig()

    # ── Expert ──
    print("Recording: Expert")
    expert = ExpertPolicy(config=env_cfg)
    expert_info = record_and_save_episodes(
        expert,
        output_dir / "expert.gif",
        num_episodes=args.num_episodes,
        env_config=env_cfg,
        fps=args.fps,
        scale=args.scale,
    )
    _print_summary("Expert", expert_info)

    # ── Random ──
    print("Recording: Random")
    random_pol = RandomPolicy(config=env_cfg)
    random_info = record_and_save_episodes(
        random_pol,
        output_dir / "random.gif",
        num_episodes=args.num_episodes,
        env_config=env_cfg,
        fps=args.fps,
        scale=args.scale,
    )
    _print_summary("Random", random_info)

    # ── π0 ──
    print(f"Recording: π0 (from {args.checkpoint})")
    model, model_cfg, flow_cfg = Trainer.load_model_from_checkpoint(
        args.checkpoint, device=args.device
    )
    pi0_policy = Pi0Policy(model, model_cfg, flow_cfg, device=args.device)
    pi0_info = record_and_save_episodes(
        pi0_policy,
        output_dir / "pi0.gif",
        num_episodes=args.num_episodes,
        env_config=env_cfg,
        fps=args.fps,
        scale=args.scale,
    )
    _print_summary("Pi0", pi0_info)

    print(f"\nGIFs saved to {output_dir}/")


def _print_summary(name: str, infos: list[dict]):
    successes = sum(1 for i in infos if i["reached"])
    avg_len = sum(i["length"] for i in infos) / len(infos)
    print(f"  {name}: {successes}/{len(infos)} reached, avg length {avg_len:.0f}")


if __name__ == "__main__":
    main()
