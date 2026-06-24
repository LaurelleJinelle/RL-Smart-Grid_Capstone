"""
train_multiseed.py
==================

Train PPO with multiple random seeds, evaluate each, and report mean and
standard deviation of performance across seeds. Used to verify that a
strong result from a single training run is reproducible -- if std across
seeds is small relative to the gap over the best baseline, the result is
solid; if std is large, training is unstable.

Saves one model per seed (ppo_seed{N}.zip) so each can be re-evaluated
later without retraining.

Requires:  pip install gymnasium stable-baselines3 torch

Run:
  python train_multiseed.py --seeds 0 1 2 3 --timesteps 200000

Resume an interrupted multi-seed run (skips seeds whose model exists):
  python train_multiseed.py --seeds 0 1 2 3 --timesteps 200000 --skip-existing
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
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--timesteps", default=None, type=int,
                        help="Per-seed training budget. Defaults: 200k for "
                             "fixed scarcity, 300k for --randomize-scarcity.")
    parser.add_argument("--eval-episodes", default=100, type=int)
    parser.add_argument("--out-dir", default=None, type=Path,
                        help="Defaults to ./output/multiseed (fixed) or "
                             "./output/multiseed_robust (randomized).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Reuse existing model files instead of retraining")
    # --- robust (domain-randomized) training flags ---
    parser.add_argument("--randomize-scarcity", action="store_true",
                        help="Train with domain randomization over scarcity "
                             "(the robust model setup).")
    parser.add_argument("--scarcity-min", default=0.4, type=float,
                        help="Lower bound of training scarcity range "
                             "(only used with --randomize-scarcity).")
    parser.add_argument("--scarcity-max", default=0.9, type=float,
                        help="Upper bound of training scarcity range "
                             "(only used with --randomize-scarcity).")
    args = parser.parse_args()

    # Smart defaults for timesteps and out_dir based on training mode.
    if args.timesteps is None:
        args.timesteps = 300_000 if args.randomize_scarcity else 200_000
    if args.out_dir is None:
        args.out_dir = Path("./output/multiseed_robust") if args.randomize_scarcity \
                       else Path("./output/multiseed")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    pilot = pd.read_csv(args.pilot)
    # Training config: respects --randomize-scarcity if set.
    train_cfg = SimConfig(
        randomize_scarcity=args.randomize_scarcity,
        scarcity_min=args.scarcity_min,
        scarcity_max=args.scarcity_max,
    )
    # Evaluation config: always FIXED scarcity 0.7 so seed-to-seed comparison
    # is apples-to-apples. Stress-testing across scarcity levels is what
    # stress_test.py is for, run separately per saved model if you want it.
    eval_cfg = SimConfig()

    mode = "RANDOMIZED-SCARCITY (robust)" if args.randomize_scarcity else "FIXED-SCARCITY"
    print(f"Mode: {mode}")
    if args.randomize_scarcity:
        print(f"  Training scarcity range: "
              f"[{args.scarcity_min}, {args.scarcity_max}]")
    print(f"  Timesteps per seed: {args.timesteps:,}")
    print(f"  Output dir: {args.out_dir}")
    print(f"  Evaluating each model at fixed scarcity 0.7\n")

    # Model file prefix so robust runs don't collide with fixed-scarcity runs
    # even if the user points them at the same out_dir.
    prefix = "ppo_robust_seed" if args.randomize_scarcity else "ppo_seed"

    per_seed_rows = []
    for seed in args.seeds:
        model_path = args.out_dir / f"{prefix}{seed}"
        zip_path = model_path.with_suffix(".zip")

        if args.skip_existing and zip_path.exists():
            print(f"[seed {seed}] reusing existing model {zip_path}")
            model = PPO.load(model_path)
        else:
            print(f"[seed {seed}] training for {args.timesteps:,} timesteps")
            env = Monitor(make_gym_env(pilot, config=train_cfg, seed=seed))
            model = PPO(
                "MlpPolicy", env,
                learning_rate=3e-4,
                n_steps=2048, batch_size=64,
                gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.01,
                verbose=0, seed=seed,
            )
            model.learn(total_timesteps=args.timesteps)
            model.save(model_path)
            print(f"[seed {seed}] saved to {zip_path}")

        # Evaluate at fixed scarcity 0.7 so seeds are comparable to each other
        # and to the existing single-run results.
        sim_eval = FeederSimulator(pilot, config=eval_cfg)
        policy = make_ppo_policy(model, sim_eval)
        stats = evaluate_policy(sim_eval, policy,
                                n_episodes=args.eval_episodes,
                                base_seed=1000 + seed * 1000)
        per_seed_rows.append({
            "seed": seed,
            "mean_reward": stats["mean_reward"],
            "std_reward": stats["std_reward"],
            "n_compliant": float(stats["compliance_rate"].sum()),
            "fairness": stats["fairness_index"],
            "total_unserved_MWh": float(stats["mean_unserved_MWh"].sum()),
        })
        print(f"[seed {seed}] mean_reward={stats['mean_reward']:.3f}, "
              f"compliant={stats['compliance_rate'].sum():.2f}, "
              f"fairness={stats['fairness_index']:.3f}\n")

    per_seed = pd.DataFrame(per_seed_rows)
    per_seed.to_csv(args.out_dir / "per_seed_results.csv", index=False)

    # Across-seed aggregation: the mean tells you the typical performance,
    # the std tells you how much training-seed luck contributes.
    metrics = ["mean_reward", "n_compliant", "fairness", "total_unserved_MWh"]
    summary = pd.DataFrame({
        "metric": metrics,
        "mean_across_seeds": [per_seed[m].mean() for m in metrics],
        "std_across_seeds": [per_seed[m].std() for m in metrics],
        "min_across_seeds": [per_seed[m].min() for m in metrics],
        "max_across_seeds": [per_seed[m].max() for m in metrics],
    })
    summary.to_csv(args.out_dir / "across_seed_summary.csv", index=False)

    # Reference baseline: demand_proportional under the SAME fixed-scarcity
    # evaluation conditions used for the PPO seeds above.
    sim_ref = FeederSimulator(pilot, config=eval_cfg)
    dp_stats = evaluate_policy(sim_ref, BASELINES["demand_proportional"],
                               n_episodes=args.eval_episodes)

    print("=" * 70)
    print("Per-seed results")
    print("=" * 70)
    print(per_seed.to_string(index=False))
    print()
    print("=" * 70)
    print("Across-seed summary (PPO)")
    print("=" * 70)
    print(summary.to_string(index=False))
    print()
    print(f"Reference: demand_proportional under same config ->")
    print(f"  mean_reward = {dp_stats['mean_reward']:.3f}, "
          f"n_compliant = {dp_stats['compliance_rate'].sum():.2f}, "
          f"fairness = {dp_stats['fairness_index']:.3f}")
    print()
    gap = per_seed["mean_reward"].mean() - dp_stats["mean_reward"]
    std_gap = per_seed["mean_reward"].std()
    print(f"PPO advantage over demand_proportional: "
          f"{gap:+.3f} (across-seed std: {std_gap:.3f})")
    if abs(gap) > 2 * std_gap:
        print("  -> Gap is large compared to seed-variance: result is solid.")
    else:
        print("  -> Gap is comparable to seed-variance: needs more seeds or "
              "longer training before drawing conclusions.")


if __name__ == "__main__":
    main()