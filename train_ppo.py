"""
Train a PPO agent on the 5-feeder environment and compare against the
baselines.

Requires extra packages not used by the simulator itself:
  pip install gymnasium stable-baselines3 torch

Run:
  python train_ppo.py --pilot ./output/pilot_feeders.csv --timesteps 200000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from feeder_env import FeederSimulator, SimConfig, BASELINES, make_gym_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", default="./output/pilot_feeders.csv", type=Path)
    parser.add_argument("--timesteps", default=200_000, type=int)
    parser.add_argument("--eval-episodes", default=100, type=int)
    parser.add_argument("--out-dir", default="./output/pilot_run", type=Path)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    # Imports here so the rest of the project doesn't depend on them.
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    pilot = pd.read_csv(args.pilot)
    cfg = SimConfig()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Training environment
    env = make_gym_env(pilot, config=cfg, seed=args.seed)
    env = Monitor(env)

    print(f"Training PPO for {args.timesteps:,} timesteps "
          f"(~{args.timesteps // cfg.steps_per_episode:,} episodes)")
    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=2048, batch_size=64,
        gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.01,
        verbose=0, seed=args.seed,
    )
    model.learn(total_timesteps=args.timesteps)
    model.save(args.out_dir / "ppo_5feeder")
    print(f"Model saved to {args.out_dir / 'ppo_5feeder'}.zip")

    # ---- evaluation ----
    print(f"\nEvaluating PPO over {args.eval_episodes} episodes")

    def ppo_policy_fn(obs_dict):
        n = len(obs_dict["band_priority"])
        steps = cfg.steps_per_episode
        flat = np.concatenate([
            obs_dict["current_demand_estimate"] / max(pilot["cap_mean"].max() / cfg.hours_per_month, 1e-6),
            obs_dict["hours_served_so_far"] / steps,
            obs_dict["committed_hours"] / steps,
            obs_dict["band_priority"] / 5.0,
            np.array([obs_dict["hour"] / steps,
                      cfg.scarcity_factor * 1.0]),
        ]).astype(np.float32)
        action, _ = model.predict(flat, deterministic=True)
        return action

    rows = []
    policies = {"ppo": ppo_policy_fn, **BASELINES}
    for name, policy in policies.items():
        sim = FeederSimulator(pilot, config=cfg)
        rewards = []
        compliance = []
        unserved = []
        for ep in range(args.eval_episodes):
            info = sim.run_episode(policy, seed=args.seed + ep)
            rewards.append(info["episode_reward"])
            compliance.append(
                (info["hours_served"] >= info["committed_hours"]).astype(float).sum()
            )
            unserved.append(info["total_unserved"].sum())
        rows.append({
            "policy": name,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_compliant_feeders": float(np.mean(compliance)),
            "mean_total_unserved": float(np.mean(unserved)),
        })

    summary = pd.DataFrame(rows).sort_values("mean_reward", ascending=False)
    summary.to_csv(args.out_dir / "ppo_vs_baselines.csv", index=False)
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
