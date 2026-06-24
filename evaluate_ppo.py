"""
evaluate_ppo.py
===============

Load a trained PPO model and evaluate it against the baselines with the
full per-feeder breakdown that `run_pilot.py` produces for the baselines.

Use this to understand HOW PPO is winning -- which feeders it serves
better, where it differs from demand_proportional, etc. The trained model
is reused, no retraining required.

Requires:  pip install gymnasium stable-baselines3 torch

Run:
  python evaluate_ppo.py --model ./output/pilot_run/ppo_5feeder.zip
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from feeder_env import FeederSimulator, SimConfig, BASELINES
from run_pilot import evaluate_policy


def make_ppo_policy(model, sim: FeederSimulator):
    """Wrap an SB3 model as a policy(obs_dict) -> raw_scores function.

    Mirrors the observation flattening in FeederAllocationEnv._flatten so
    the model receives observations in exactly the same shape as during
    training. Uses obs_dict["supply_target"] rather than cfg.scarcity_factor
    so it stays correct under stress-test conditions (different scarcity).
    """
    base_max = max(sim.base_demand.max(), 1e-6)
    base_sum = max(sim.base_demand.sum(), 1e-6)
    steps = sim.cfg.steps_per_episode

    def policy_fn(obs_dict):
        flat = np.concatenate([
            obs_dict["current_demand_estimate"] / base_max,
            obs_dict["hours_served_so_far"] / steps,
            obs_dict["committed_hours"] / steps,
            obs_dict["band_priority"] / 5.0,
            np.array([obs_dict["hour"] / steps,
                      obs_dict["supply_target"] / base_sum]),
        ]).astype(np.float32)
        action, _ = model.predict(flat, deterministic=True)
        return action

    return policy_fn


def print_per_feeder(name: str, pilot: pd.DataFrame, sim: FeederSimulator, stats: dict):
    df = pd.DataFrame({
        "band": pilot["band"].values,
        "committed_hrs": sim.committed_hours,
        "served_hrs_mean": stats["mean_hours_served"].round(2),
        "compliance_rate": stats["compliance_rate"].round(2),
        "delivered_MWh": stats["mean_delivered_MWh"].round(3),
        "unserved_MWh": stats["mean_unserved_MWh"].round(3),
    })
    print(f"--- {name} ---")
    print(df.to_string(index=False))
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path,
                        help="Path to saved PPO model (.zip)")
    parser.add_argument("--pilot", default="./output/pilot_feeders.csv", type=Path)
    parser.add_argument("--episodes", default=100, type=int)
    parser.add_argument("--out-dir", default="./output/ppo_eval", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from stable_baselines3 import PPO

    pilot = pd.read_csv(args.pilot)
    cfg = SimConfig()
    # SB3 accepts paths with or without the .zip suffix.
    model = PPO.load(str(args.model).removesuffix(".zip"))
    print(f"Loaded model from {args.model}")
    print(f"Evaluating over {args.episodes} episodes\n")

    sim_for_ppo = FeederSimulator(pilot, config=cfg)
    policies = {
        "ppo": make_ppo_policy(model, sim_for_ppo),
        **BASELINES,
    }

    rows = []
    for name, policy in policies.items():
        sim = FeederSimulator(pilot, config=cfg)
        stats = evaluate_policy(sim, policy, n_episodes=args.episodes)
        print_per_feeder(name, pilot, sim, stats)
        rows.append({
            "policy": name,
            "mean_reward": stats["mean_reward"],
            "std_reward": stats["std_reward"],
            "n_compliant_mean": float(stats["compliance_rate"].sum()),
            "fairness": stats["fairness_index"],
            "total_unserved_MWh": float(stats["mean_unserved_MWh"].sum()),
        })

    summary = pd.DataFrame(rows).sort_values("mean_reward", ascending=False)
    summary.to_csv(args.out_dir / "ppo_per_feeder_eval.csv", index=False)
    print("=" * 70)
    print("Summary (sorted by mean_reward)")
    print("=" * 70)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
