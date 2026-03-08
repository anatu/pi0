"""CLI script to collect expert demonstration trajectories."""

import argparse

from pi0.config import EnvConfig
from pi0.data.collector import collect_trajectories


def main():
    parser = argparse.ArgumentParser(description="Collect expert trajectories")
    parser.add_argument("--num_trajectories", type=int, default=2000)
    parser.add_argument("--output_dir", type=str, default="data/trajectories")
    parser.add_argument("--noise_std", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    collect_trajectories(
        num_trajectories=args.num_trajectories,
        output_dir=args.output_dir,
        env_config=EnvConfig(),
        expert_noise_std=args.noise_std,
        seed=args.seed,
    )
    print(f"Collected {args.num_trajectories} trajectories in {args.output_dir}")


if __name__ == "__main__":
    main()
