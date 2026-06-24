"""
Simulator and Gymnasium environment for the IKEDC 5-feeder pilot.

Two classes:
  - FeederSimulator   pure-NumPy simulation logic, no external deps beyond
                      numpy and pandas. Use this to test policies, run
                      baselines, or build evaluation scripts.
  - FeederAllocationEnv  thin Gymnasium wrapper around FeederSimulator,
                         imported lazily so this file can be loaded even
                         when gymnasium is not installed.

Concepts
--------
* One episode = one day (24 hourly steps).
* Each step:
    1. Demand for each feeder is drawn (load shape * base demand * noise).
    2. Total supply for the hour is drawn (target supply * noise, with rare
       shortages).
    3. The agent outputs raw scores; we softmax them into allocation
       shares that sum to 1.0.
    4. delivered[i]   = min(share[i] * supply, demand[i])
       unserved[i]    = demand[i] - delivered[i]
       wasted[i]      = share[i] * supply - delivered[i]
    5. A feeder is counted as "served" this hour if delivered/demand >= 0.5
       (or if demand is near zero). hours_served accumulates over the day.
    6. Per-step reward = -unserved (weighted by band priority) - 0.2 * wasted,
       normalized by mean total demand.
* End of episode:
    + bonus for each feeder whose hours_served met its committed_hours
    + Jain's fairness index on (hours_served / committed_hours) across feeders
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# Lagos-like 24-hour load shape (peaks at 7-9am and 6-9pm; quiet 1-5am).
# Multiplies base demand. Mean over 24h is ~1.0 by construction.
DEFAULT_LOAD_SHAPE = np.array([
    0.65, 0.60, 0.58, 0.58, 0.60, 0.70,   # 0-5
    0.95, 1.20, 1.35, 1.25, 1.15, 1.10,   # 6-11
    1.10, 1.05, 1.00, 1.00, 1.05, 1.20,   # 12-17
    1.40, 1.45, 1.35, 1.20, 1.00, 0.80,   # 18-23
])
DEFAULT_LOAD_SHAPE = DEFAULT_LOAD_SHAPE / DEFAULT_LOAD_SHAPE.mean()


@dataclass
class SimConfig:
    """Tunable parameters for the simulator."""
    scarcity_factor: float = 0.7         # supply = scarcity * mean(total_demand)
    demand_noise_pct: float = 0.10       # gaussian noise on demand per step
    supply_noise_pct: float = 0.10       # gaussian noise on supply per step
    shortage_prob: float = 0.05          # probability of a "shortage" hour
    shortage_factor: float = 0.4         # supply multiplier during shortage
    served_threshold: float = 0.5        # delivered/demand >= this counts
    min_demand_for_served: float = 0.01  # if demand below this, count served
    unserved_priority_weight: bool = False  # band priority is already in committed_hours
    wasted_penalty: float = 0.2          # coefficient on wasted energy
    compliance_bonus: float = 2.0        # per feeder meeting committed hours
    shortfall_penalty_weight: float = 0.5  # per hour missed below committed
    fairness_bonus_weight: float = 3.0   # weight on Jain's fairness index
    steps_per_episode: int = 24
    hours_per_month: int = 720           # 30 days * 24h; for cap-to-MWh/h conv
    # Domain randomization: redraw scarcity each episode if True.
    # When True, scarcity_factor above is IGNORED in favor of a uniform draw
    # from [scarcity_min, scarcity_max] at every reset(). The agent can observe
    # the resulting supply_target so it can adapt its policy to the current
    # scarcity level instead of overfitting to one.
    randomize_scarcity: bool = False
    scarcity_min: float = 0.4
    scarcity_max: float = 0.9


@dataclass
class StepInfo:
    """Diagnostic fields returned alongside each step."""
    demand: np.ndarray
    supply: float
    allocation_share: np.ndarray
    delivered: np.ndarray
    unserved: np.ndarray
    wasted: np.ndarray
    served_flag: np.ndarray
    is_shortage: bool


class FeederSimulator:
    """Pure-NumPy simulation of N feeders sharing a constrained supply."""

    def __init__(
        self,
        feeders_df: pd.DataFrame,
        config: Optional[SimConfig] = None,
        load_shape: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ):
        required = ["feeder", "band", "band_priority", "committed_hours_per_day", "cap_mean"]
        missing = [c for c in required if c not in feeders_df.columns]
        if missing:
            raise ValueError(f"feeders_df missing columns: {missing}")

        self.cfg = config or SimConfig()
        self.load_shape = load_shape if load_shape is not None else DEFAULT_LOAD_SHAPE
        if len(self.load_shape) != self.cfg.steps_per_episode:
            raise ValueError(
                f"load_shape length {len(self.load_shape)} does not match "
                f"steps_per_episode {self.cfg.steps_per_episode}"
            )

        self.feeders = feeders_df.reset_index(drop=True).copy()
        self.n = len(self.feeders)
        self.feeder_names = self.feeders["feeder"].tolist()
        self.bands = self.feeders["band"].tolist()
        self.band_priority = self.feeders["band_priority"].to_numpy(dtype=float)
        self.committed_hours = self.feeders["committed_hours_per_day"].to_numpy(dtype=float)

        # Convert monthly cap (MWh) to average MWh-per-hour for that feeder.
        self.base_demand = (
            self.feeders["cap_mean"].to_numpy(dtype=float) / self.cfg.hours_per_month
        )

        # Pre-compute reward normalization: divide by mean total demand so per-
        # step reward magnitude is roughly in [-1, 0]. Doesn't depend on
        # scarcity so it stays constant under domain randomization.
        self._reward_norm = max(self.base_demand.sum(), 1e-6)

        self.rng = np.random.default_rng(seed)
        # supply_target is set per-episode in reset() so randomize_scarcity
        # can redraw it. Initialized here so attribute always exists.
        self.supply_target = self.cfg.scarcity_factor * self.base_demand.sum()
        self.reset()

    # ---- core API ----

    def reset(self, seed: Optional[int] = None) -> dict:
        """Start a fresh episode and return the initial observation dict."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # Redraw the scarcity factor for this episode if randomization is on.
        # The agent sees the resulting supply_target in its observation, so
        # it can adapt allocation strategy to current scarcity.
        if self.cfg.randomize_scarcity:
            scarcity = self.rng.uniform(self.cfg.scarcity_min, self.cfg.scarcity_max)
        else:
            scarcity = self.cfg.scarcity_factor
        self.supply_target = scarcity * self.base_demand.sum()
        self.current_scarcity = scarcity  # exposed for diagnostics
        self.step_idx = 0
        self.hours_served = np.zeros(self.n, dtype=float)
        self.total_delivered = np.zeros(self.n, dtype=float)
        self.total_unserved = np.zeros(self.n, dtype=float)
        self.total_wasted = np.zeros(self.n, dtype=float)
        self.history: list[StepInfo] = []
        return self._observation()

    def step(self, raw_scores: np.ndarray) -> tuple[dict, float, bool, dict]:
        """Advance one hour. raw_scores has shape (n_feeders,)."""
        if self.step_idx >= self.cfg.steps_per_episode:
            raise RuntimeError("episode already finished; call reset()")

        hour = self.step_idx
        demand = self._draw_demand(hour)
        supply, is_shortage = self._draw_supply()
        shares = self._softmax(np.asarray(raw_scores, dtype=float))
        attempted = shares * supply
        delivered = np.minimum(attempted, demand)
        unserved = demand - delivered
        wasted = attempted - delivered

        served_flag = (
            (delivered >= self.cfg.served_threshold * demand)
            | (demand < self.cfg.min_demand_for_served)
        ).astype(float)
        self.hours_served += served_flag
        self.total_delivered += delivered
        self.total_unserved += unserved
        self.total_wasted += wasted

        reward = self._step_reward(unserved, wasted)
        self.history.append(StepInfo(
            demand=demand, supply=supply, allocation_share=shares,
            delivered=delivered, unserved=unserved, wasted=wasted,
            served_flag=served_flag, is_shortage=is_shortage,
        ))
        self.step_idx += 1

        done = self.step_idx >= self.cfg.steps_per_episode
        if done:
            reward += self._end_of_episode_reward()

        obs = self._observation()
        info = self._info(reward, done)
        return obs, float(reward), done, info

    # ---- internals ----

    def _draw_demand(self, hour: int) -> np.ndarray:
        shape_mult = self.load_shape[hour]
        noise = self.rng.normal(1.0, self.cfg.demand_noise_pct, size=self.n)
        noise = np.clip(noise, 0.3, 2.0)
        return self.base_demand * shape_mult * noise

    def _draw_supply(self) -> tuple[float, bool]:
        is_shortage = bool(self.rng.random() < self.cfg.shortage_prob)
        mult = self.cfg.shortage_factor if is_shortage else 1.0
        noise = float(self.rng.normal(1.0, self.cfg.supply_noise_pct))
        noise = float(np.clip(noise, 0.3, 1.5))
        return max(self.supply_target * mult * noise, 1e-6), is_shortage

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()

    def _step_reward(self, unserved: np.ndarray, wasted: np.ndarray) -> float:
        if self.cfg.unserved_priority_weight:
            unserved_term = float((unserved * self.band_priority).sum())
        else:
            unserved_term = float(unserved.sum())
        wasted_term = float(wasted.sum())
        return -(unserved_term + self.cfg.wasted_penalty * wasted_term) / self._reward_norm

    def _end_of_episode_reward(self) -> float:
        # Binary compliance bonus: one per feeder that met its committed hours.
        met = (self.hours_served >= self.committed_hours).astype(float)
        compliance = self.cfg.compliance_bonus * float(met.sum())
        # Linear shortfall penalty: hurts proportionally to how many committed
        # hours each feeder fell short by. Gives PPO a gradient to climb even
        # when no feeder has crossed the compliance threshold yet.
        shortfall = np.maximum(self.committed_hours - self.hours_served, 0.0)
        shortfall_penalty = self.cfg.shortfall_penalty_weight * float(shortfall.sum())
        # Jain's fairness on hours_served / committed_hours.
        ratios = self.hours_served / np.maximum(self.committed_hours, 1e-6)
        ratios = np.clip(ratios, 0, 2.0)
        jain = (ratios.sum() ** 2) / (self.n * (ratios ** 2).sum() + 1e-9)
        return compliance - shortfall_penalty + self.cfg.fairness_bonus_weight * float(jain)

    def _observation(self) -> dict:
        hour = self.step_idx if self.step_idx < self.cfg.steps_per_episode else self.cfg.steps_per_episode - 1
        current_demand = self.base_demand * self.load_shape[hour]  # noise-free estimate
        return {
            "hour": hour,
            "step_idx": self.step_idx,
            "current_demand_estimate": current_demand,
            "hours_served_so_far": self.hours_served.copy(),
            "committed_hours": self.committed_hours.copy(),
            "band_priority": self.band_priority.copy(),
            "supply_target": self.supply_target,
        }

    def _info(self, reward: float, done: bool) -> dict:
        last = self.history[-1] if self.history else None
        return {
            "reward": reward,
            "done": done,
            "hours_served": self.hours_served.copy(),
            "committed_hours": self.committed_hours.copy(),
            "total_delivered": self.total_delivered.copy(),
            "total_unserved": self.total_unserved.copy(),
            "total_wasted": self.total_wasted.copy(),
            "last_step": last,
        }

    # ---- convenience for evaluation ----

    def run_episode(self, policy_fn, seed: Optional[int] = None) -> dict:
        """Run one full episode with a callable policy(obs) -> raw_scores."""
        obs = self.reset(seed=seed)
        total_reward = 0.0
        while True:
            action = policy_fn(obs)
            obs, reward, done, info = self.step(action)
            total_reward += reward
            if done:
                break
        info["episode_reward"] = total_reward
        return info


# --------------------------------------------------------------------------
# Built-in baseline policies
# --------------------------------------------------------------------------

def equal_split_policy(obs: dict) -> np.ndarray:
    """Allocate equal shares to all feeders, regardless of state."""
    n = len(obs["band_priority"])
    return np.zeros(n)  # equal raw scores -> softmax to 1/n each


def priority_based_policy(obs: dict) -> np.ndarray:
    """Allocate in proportion to band priority. Highest band first."""
    return obs["band_priority"].astype(float)


def demand_proportional_policy(obs: dict) -> np.ndarray:
    """Allocate in proportion to current demand estimate."""
    d = obs["current_demand_estimate"]
    # Take log so softmax preserves proportionality.
    return np.log(d + 1e-6)


def deficit_chasing_policy(obs: dict) -> np.ndarray:
    """Give more to feeders that are behind on committed hours so far."""
    deficit = obs["committed_hours"] - obs["hours_served_so_far"]
    deficit = np.maximum(deficit, 0)
    return deficit * obs["band_priority"]


BASELINES = {
    "equal_split": equal_split_policy,
    "priority_based": priority_based_policy,
    "demand_proportional": demand_proportional_policy,
    "deficit_chasing": deficit_chasing_policy,
}


# --------------------------------------------------------------------------
# Gymnasium wrapper (lazy import so this file works without gymnasium)
# --------------------------------------------------------------------------

def make_gym_env(feeders_df: pd.DataFrame, config: Optional[SimConfig] = None,
                 seed: Optional[int] = None):
    """Return a Gymnasium-compatible env wrapping FeederSimulator.

    Imported lazily so this module can be used without gymnasium installed.
    """
    import gymnasium as gym
    from gymnasium import spaces

    class FeederAllocationEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.sim = FeederSimulator(feeders_df, config=config, seed=seed)
            n = self.sim.n
            # Action: 5 unbounded raw scores; the env softmaxes them.
            self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(n,), dtype=np.float32)
            # Observation vector: per feeder (demand_est, hours_served/24,
            # committed/24, priority/5), plus (hour/24, supply_target_norm).
            obs_dim = n * 4 + 2
            self.observation_space = spaces.Box(
                low=0.0, high=10.0, shape=(obs_dim,), dtype=np.float32
            )

        def _flatten(self, obs_dict):
            n = self.sim.n
            steps = self.sim.cfg.steps_per_episode
            flat = np.concatenate([
                obs_dict["current_demand_estimate"] / max(self.sim.base_demand.max(), 1e-6),
                obs_dict["hours_served_so_far"] / steps,
                obs_dict["committed_hours"] / steps,
                obs_dict["band_priority"] / 5.0,
                np.array([obs_dict["hour"] / steps,
                          obs_dict["supply_target"] / max(self.sim.base_demand.sum(), 1e-6)]),
            ]).astype(np.float32)
            return flat

        def reset(self, *, seed=None, options=None):
            obs_dict = self.sim.reset(seed=seed)
            return self._flatten(obs_dict), {}

        def step(self, action):
            obs_dict, reward, done, info = self.sim.step(np.asarray(action))
            return self._flatten(obs_dict), reward, done, False, info

    return FeederAllocationEnv()