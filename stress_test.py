"""
stress_test.py
==============

Evaluate a trained PPO model under scarcity and shortage conditions that
are HARDER than what it was trained on (training was scarcity=0.7,
shortage_prob=0.05). Compares against demand_proportional under the same
conditions to show how each policy degrades.

This is the robustness experiment: if PPO holds up better than the static
demand_proportional heuristic when conditions worsen, that is a stronger
finding than the within-distribution win.

Requires:  pip install gymnasium stable-baselines3 torch

Run:
  python stress_test.py --model ./output/pilot_run/ppo_5feeder.zip
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from feeder_env import FeederSimulator, SimConfig, BASELINES
from run_pilot import evaluate_policy
from evaluate_ppo import make_ppo_policy


# (label, scarcity_factor, shortage_prob, shortage_factor)
#   scarcity_factor = mean supply / mean total demand (lower = harder)
#   shortage_prob   = probability any given hour is a shortage event
#   shortage_factor = supply multiplier during a shortage event
STRESS_LEVELS = [
    ("easy",     0.90, 0.02, 0.5),
    ("baseline", 0.70, 0.05, 0.4),   # matches training conditions
    ("hard",     0.50, 0.10, 0.4),
    ("extreme",  0.30, 0.15, 0.3),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path,
                        help="Path to saved PPO model (.zip)")
    parser.add_argument("--pilot", default="./output/pilot_feeders.csv", type=Path)
    parser.add_argument("--episodes", default=100, type=int)
    parser.add_argument("--out-dir", default="./output/stress_test", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from stable_baselines3 import PPO

    pilot = pd.read_csv(args.pilot)
    model = PPO.load(str(args.model).removesuffix(".zip"))
    print(f"Loaded model from {args.model}")
    print(f"Stress-testing across {len(STRESS_LEVELS)} scenarios, "
          f"{args.episodes} episodes each\n")

    rows = []
    for label, scarcity, shortage_p, shortage_f in STRESS_LEVELS:
        cfg = SimConfig(
            scarcity_factor=scarcity,
            shortage_prob=shortage_p,
            shortage_factor=shortage_f,
        )
        # Build a sim once for the policy_fn closure (it captures normalizers
        # from sim.base_demand, which depend only on the pilot data, not cfg).
        sim_for_ppo = FeederSimulator(pilot, config=cfg)
        ppo_policy = make_ppo_policy(model, sim_for_ppo)

        # Evaluate PPO and the strongest baseline under this scenario.
        for name, policy in [
            ("ppo", ppo_policy),
            ("demand_proportional", BASELINES["demand_proportional"]),
        ]:
            sim = FeederSimulator(pilot, config=cfg)
            stats = evaluate_policy(sim, policy,
                                    n_episodes=args.episodes, base_seed=42)
            rows.append({
                "scenario": label,
                "scarcity": scarcity,
                "shortage_prob": shortage_p,
                "policy": name,
                "mean_reward": stats["mean_reward"],
                "std_reward": stats["std_reward"],
                "n_compliant": float(stats["compliance_rate"].sum()),
                "fairness": stats["fairness_index"],
                "total_unserved_MWh": float(stats["mean_unserved_MWh"].sum()),
            })
            print(f"  [{label:<10}] {name:<22} "
                  f"reward={stats['mean_reward']:+7.3f}, "
                  f"compliant={stats['compliance_rate'].sum():.2f}/5, "
                  f"unserved={stats['mean_unserved_MWh'].sum():.1f} MWh")
        print()

    results = pd.DataFrame(rows)
    results.to_csv(args.out_dir / "stress_test_results.csv", index=False)

    # Pivot to show degradation curve.
    print("=" * 70)
    print("Reward across scenarios (and PPO advantage)")
    print("=" * 70)
    pivot = results.pivot(index="scenario", columns="policy",
                          values="mean_reward")
    pivot = pivot.reindex([s[0] for s in STRESS_LEVELS])
    pivot["ppo_advantage"] = pivot["ppo"] - pivot["demand_proportional"]
    print(pivot.round(3).to_string())
    print()

    print("Compliance (out of 5) across scenarios:")
    pivot_c = results.pivot(index="scenario", columns="policy",
                            values="n_compliant")
    pivot_c = pivot_c.reindex([s[0] for s in STRESS_LEVELS])
    print(pivot_c.round(2).to_string())


if __name__ == "__main__":
    main()