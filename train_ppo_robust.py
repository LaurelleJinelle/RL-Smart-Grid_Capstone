"""
train_ppo_robust.py
===================

Train PPO with DOMAIN RANDOMIZATION over the scarcity factor. Each episode
draws a new scarcity uniformly from [scarcity_min, scarcity_max], so the
agent must learn one policy that adapts to any supply level rather than
one that only works at scarcity 0.7.

The agent already observes its current supply level (it's the last element
of the flattened observation vector), so it has the information needed to
condition its strategy on the current scarcity. With training, it should
learn to use that information.

This is the recommended fix for the distribution-shift problem revealed by
stress_test.py: PPO trained at fixed scarcity 0.7 has lower compliance
than demand-proportional under scarcity 0.5.

Requires:  pip install gymnasium stable-baselines3 torch

Run:
  python train_ppo_robust.py --timesteps 300000

Then test it the same way as the original model:
  python stress_test.py --model ./output/pilot_run/ppo_5feeder_robust.zip
  python evaluate_ppo.py --model ./output/pilot_run/ppo_5feeder_robust.zip
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from feeder_env import FeederSimulator, SimConfig, BASELINES, make_gym_env
from run_pilot import evaluate_policy
from evaluate_ppo import make_ppo_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", default="./output/pilot_feeders.csv", type=Path)
    parser.add_argument("--timesteps", default=300_000, type=int,
                        help="More than 200k recommended; the agent must learn "
                             "multiple scarcity regimes, not just one.")
    parser.add_argument("--scarcity-min", default=0.4, type=float,
                        help="Lowest scarcity drawn during training (harder).")
    parser.add_argument("--scarcity-max", default=0.9, type=float,
                        help="Highest scarcity drawn during training (easier).")
    parser.add_argument("--eval-episodes", default=100, type=int)
    parser.add_argument("--out-dir", default="./output/pilot_run", type=Path)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    pilot = pd.read_csv(args.pilot)

    train_cfg = SimConfig(
        randomize_scarcity=True,
        scarcity_min=args.scarcity_min,
        scarcity_max=args.scarcity_max,
    )
    print(f"Training PPO with randomized scarcity in "
          f"[{args.scarcity_min}, {args.scarcity_max}]")
    print(f"Timesteps: {args.timesteps:,} "
          f"(~{args.timesteps // train_cfg.steps_per_episode:,} episodes)\n")

    env = Monitor(make_gym_env(pilot, config=train_cfg, seed=args.seed))
    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=2048, batch_size=64,
        gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.01,
        verbose=0, seed=args.seed,
    )
    model.learn(total_timesteps=args.timesteps)
    save_path = args.out_dir / "ppo_5feeder_robust"
    model.save(save_path)
    print(f"\nModel saved to {save_path}.zip")

    # ---- evaluate at fixed scarcity 0.7 for direct comparison ----
    print("\nQuick evaluation at fixed scarcity 0.7 "
          "(matches original PPO's training condition)")
    eval_cfg = SimConfig()  # default scarcity 0.7, randomize_scarcity False
    sim = FeederSimulator(pilot, config=eval_cfg)
    policy = make_ppo_policy(model, sim)
    stats = evaluate_policy(sim, policy, n_episodes=args.eval_episodes)
    print(f"  robust-PPO:          reward={stats['mean_reward']:+.3f}, "
          f"compliant={stats['compliance_rate'].sum():.2f}/5, "
          f"fairness={stats['fairness_index']:.3f}")
    print(f"  original PPO (ref):  reward~+4.41, compliant~4.96/5")
    print(f"  demand_proportional: reward~+2.86, compliant~4.47/5")


if __name__ == "__main__":
    main()