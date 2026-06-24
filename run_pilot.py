"""
Run the 5-feeder pilot simulation with each baseline policy, print a
comparison table, and save a plot showing what happens over one day.

Run:
  python run_pilot.py --pilot ./output/pilot_feeders.csv --episodes 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from feeder_env import FeederSimulator, SimConfig, BASELINES


def evaluate_policy(
    sim: FeederSimulator, policy_fn, n_episodes: int = 30, base_seed: int = 0
) -> dict:
    """Run n_episodes with the given policy and return aggregate stats."""
    rewards = []
    served_records = []
    delivered_records = []
    unserved_records = []
    wasted_records = []
    compliance_records = []
    for ep in range(n_episodes):
        info = sim.run_episode(policy_fn, seed=base_seed + ep)
        rewards.append(info["episode_reward"])
        served_records.append(info["hours_served"])
        delivered_records.append(info["total_delivered"])
        unserved_records.append(info["total_unserved"])
        wasted_records.append(info["total_wasted"])
        compliance_records.append(
            (info["hours_served"] >= info["committed_hours"]).astype(float)
        )
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_hours_served": np.mean(served_records, axis=0),
        "mean_delivered_MWh": np.mean(delivered_records, axis=0),
        "mean_unserved_MWh": np.mean(unserved_records, axis=0),
        "mean_wasted_MWh": np.mean(wasted_records, axis=0),
        "compliance_rate": np.mean(compliance_records, axis=0),
        "fairness_index": _fairness(np.mean(served_records, axis=0),
                                    sim.committed_hours),
    }


def _fairness(hours_served: np.ndarray, committed: np.ndarray) -> float:
    ratios = hours_served / np.maximum(committed, 1e-6)
    ratios = np.clip(ratios, 0, 2.0)
    return float((ratios.sum() ** 2) / (len(ratios) * (ratios ** 2).sum() + 1e-9))


def plot_one_episode(
    sim: FeederSimulator, policy_fn, out_path: Path, title: str, seed: int = 0
) -> None:
    """Plot demand, supply, and delivered per feeder for one episode."""
    import matplotlib.pyplot as plt

    info = sim.run_episode(policy_fn, seed=seed)
    history = sim.history
    hours = np.arange(len(history))
    demand_mat = np.stack([h.demand for h in history])         # (24, n)
    delivered_mat = np.stack([h.delivered for h in history])
    supply_vec = np.array([h.supply for h in history])

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax = axes[0]
    ax.plot(hours, demand_mat.sum(axis=1), label="Total demand", color="#993C1D")
    ax.plot(hours, supply_vec, label="Available supply", color="#185FA5")
    ax.plot(hours, delivered_mat.sum(axis=1), label="Total delivered",
            color="#0F6E56", linestyle="--")
    ax.set_ylabel("MWh / hour")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    for i, name in enumerate(sim.feeder_names):
        ax.plot(hours, delivered_mat[:, i],
                label=f"{sim.bands[i]}: {name[:30]}")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Delivered MWh / hour")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", default="./output/pilot_feeders.csv", type=Path)
    parser.add_argument("--episodes", default=30, type=int)
    parser.add_argument("--out-dir", default="./output/pilot_run", type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pilot = pd.read_csv(args.pilot)
    print(f"Loaded {len(pilot)} pilot feeders:")
    for _, r in pilot.iterrows():
        print(f"  {r['band']:<3}  {r['feeder']:<32}  cap={r['cap_mean']:.1f}  "
              f"committed_hours={r['committed_hours_per_day']:.1f}")
    print()

    cfg = SimConfig()
    print(f"Simulator config: scarcity={cfg.scarcity_factor}, "
          f"shortage_prob={cfg.shortage_prob}, episodes={args.episodes}")
    print()

    rows = []
    for name, policy in BASELINES.items():
        sim = FeederSimulator(pilot, config=cfg)
        stats = evaluate_policy(sim, policy, n_episodes=args.episodes)
        rows.append({"policy": name, **{
            "mean_reward": stats["mean_reward"],
            "fairness": stats["fairness_index"],
            "n_compliant": int(stats["compliance_rate"].sum()),  # out of 5
            "total_unserved": float(stats["mean_unserved_MWh"].sum()),
            "total_wasted": float(stats["mean_wasted_MWh"].sum()),
        }})

        # Per-feeder breakdown
        print(f"--- {name} ---")
        per_feeder = pd.DataFrame({
            "band": pilot["band"].values,
            "committed_hrs": sim.committed_hours,
            "served_hrs_mean": stats["mean_hours_served"].round(2),
            "compliance_rate": stats["compliance_rate"].round(2),
            "delivered_MWh": stats["mean_delivered_MWh"].round(3),
            "unserved_MWh": stats["mean_unserved_MWh"].round(3),
        })
        print(per_feeder.to_string(index=False))
        print()

        # Save one example-episode plot per policy
        plot_one_episode(
            FeederSimulator(pilot, config=cfg),
            policy,
            args.out_dir / f"day_plot_{name}.png",
            title=f"One-day simulation: {name}",
            seed=0,
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "policy_comparison.csv", index=False)
    print("=" * 60)
    print("Summary across baselines")
    print("=" * 60)
    print(summary.to_string(index=False))
    print()
    print(f"Plots and summary written to {args.out_dir}")


if __name__ == "__main__":
    main()
