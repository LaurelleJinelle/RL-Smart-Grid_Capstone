"""
FastAPI backend for the Smart Grid RL Dashboard.

Loads the trained PPO models at startup and exposes endpoints that the
React frontend calls to run simulations, compare policies, and stress-test.

Run from rl_pipeline/dashboard/backend/:
  uvicorn app:app --reload --port 8000
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
import os   
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Make the project root importable so we can pull in the simulator code.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from feeder_env import FeederSimulator, SimConfig, BASELINES  # noqa: E402
from run_pilot import evaluate_policy                         # noqa: E402
from evaluate_ppo import make_ppo_policy                      # noqa: E402


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

PILOT_CSV = PROJECT_ROOT / "output" / "pilot_feeders.csv"
MODEL_PATHS = {
    "ppo": PROJECT_ROOT / "output" / "pilot_run" / "ppo_5feeder.zip",
    "ppo_robust": PROJECT_ROOT / "output" / "pilot_run" / "ppo_5feeder_robust.zip",
}

# Held in module state once loaded; populated in the lifespan handler.
pilot_df: Optional[pd.DataFrame] = None
loaded_models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load CSVs and PPO models once at startup."""
    global pilot_df, loaded_models

    if not PILOT_CSV.exists():
        raise RuntimeError(
            f"Could not find pilot_feeders.csv at {PILOT_CSV}. "
            f"Make sure you've run select_pilot_feeders.py first."
        )
    pilot_df = pd.read_csv(PILOT_CSV)
    print(f"[startup] Loaded {len(pilot_df)} pilot feeders")

    # Models are optional -- the dashboard still works for baselines if none
    # of the PPO models are present, just with the PPO options hidden.
    from stable_baselines3 import PPO
    for name, path in MODEL_PATHS.items():
        if path.exists():
            loaded_models[name] = PPO.load(str(path).removesuffix(".zip"))
            print(f"[startup] Loaded model {name} from {path.name}")
        else:
            print(f"[startup] WARNING: {path} not found; '{name}' disabled")

    yield

    # No teardown needed.


app = FastAPI(title="Smart Grid RL Dashboard", lifespan=lifespan)

# In development, the React dev server runs on a different port, so CORS is
# necessary. The Vite config also proxies /api so this is belt-and-braces.
allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
# Add production frontend URL from environment variable
prod_origin = os.environ.get("FRONTEND_URL")
if prod_origin:
    allowed_origins.append(prod_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _resolve_policy(policy_name: str, sim: FeederSimulator):
    """Look up a policy by name, returning the callable or raising 400."""
    if policy_name in BASELINES:
        return BASELINES[policy_name]
    if policy_name in loaded_models:
        return make_ppo_policy(loaded_models[policy_name], sim)
    raise HTTPException(
        status_code=400,
        detail=f"Unknown or unavailable policy: {policy_name}",
    )


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

class SimulateRequest(BaseModel):
    policy: str
    scarcity: float = 0.7
    shortage_prob: float = 0.05
    seed: int = 0


class CompareRequest(BaseModel):
    policies: list[str]
    scarcity: float = 0.7
    shortage_prob: float = 0.05
    episodes: int = 30


class StressTestRequest(BaseModel):
    policies: list[str] = ["ppo_robust", "demand_proportional"]
    episodes: int = 30


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"ok": True, "models_loaded": list(loaded_models.keys())}


@app.get("/api/feeders")
def get_feeders():
    """Return the pilot feeders for display."""
    return pilot_df.to_dict(orient="records")


@app.get("/api/policies")
def get_policies():
    """Return the policies the dashboard can actually run on this server."""
    items = []
    for name in loaded_models:
        items.append({
            "id": name,
            "kind": "rl",
            "label": "PPO (Robust)" if name == "ppo_robust" else "PPO (Original)",
        })
    label_map = {
        "demand_proportional": "Demand-Proportional",
        "equal_split": "Equal Split",
        "priority_based": "Priority-Based",
        "deficit_chasing": "Deficit-Chasing",
    }
    for name in BASELINES:
        items.append({
            "id": name,
            "kind": "baseline",
            "label": label_map.get(name, name),
        })
    return items


@app.post("/api/simulate")
def simulate(req: SimulateRequest):
    """Run one episode and return hour-by-hour breakdown for visualisation."""
    cfg = SimConfig(scarcity_factor=req.scarcity, shortage_prob=req.shortage_prob)
    sim = FeederSimulator(pilot_df, config=cfg, seed=req.seed)
    policy_fn = _resolve_policy(req.policy, sim)

    obs = sim.reset(seed=req.seed)
    hourly = []
    total_reward = 0.0
    for hour in range(24):
        action = policy_fn(obs)
        obs, reward, done, info = sim.step(action)
        total_reward += reward
        last = info["last_step"]
        hourly.append({
            "hour": hour,
            "supply": float(last.supply),
            "total_demand": float(last.demand.sum()),
            "total_delivered": float(last.delivered.sum()),
            "is_shortage": bool(last.is_shortage),
            "per_feeder_demand": last.demand.tolist(),
            "per_feeder_delivered": last.delivered.tolist(),
            "per_feeder_share": last.allocation_share.tolist(),
        })

    return {
        "total_reward": float(total_reward),
        "hours_served": sim.hours_served.tolist(),
        "committed_hours": sim.committed_hours.tolist(),
        "compliant_feeders": int((sim.hours_served >= sim.committed_hours).sum()),
        "total_unserved_MWh": float(sim.total_unserved.sum()),
        "total_delivered_MWh": float(sim.total_delivered.sum()),
        "feeder_names": sim.feeder_names,
        "bands": sim.bands,
        "hourly": hourly,
    }


@app.post("/api/compare")
def compare_policies(req: CompareRequest):
    """Run N episodes for each policy, return aggregates."""
    cfg = SimConfig(scarcity_factor=req.scarcity, shortage_prob=req.shortage_prob)

    results = []
    for policy_name in req.policies:
        sim = FeederSimulator(pilot_df, config=cfg)
        try:
            policy_fn = _resolve_policy(policy_name, sim)
        except HTTPException:
            continue
        stats = evaluate_policy(sim, policy_fn, n_episodes=req.episodes)
        results.append({
            "policy": policy_name,
            "mean_reward": float(stats["mean_reward"]),
            "std_reward": float(stats["std_reward"]),
            "n_compliant": float(stats["compliance_rate"].sum()),
            "fairness": float(stats["fairness_index"]),
            "per_feeder_compliance": stats["compliance_rate"].tolist(),
            "per_feeder_served_hours": stats["mean_hours_served"].tolist(),
            "per_feeder_delivered": stats["mean_delivered_MWh"].tolist(),
            "per_feeder_unserved": stats["mean_unserved_MWh"].tolist(),
            "total_unserved_MWh": float(stats["mean_unserved_MWh"].sum()),
        })

    return {
        "feeder_names": list(pilot_df["feeder"]),
        "bands": list(pilot_df["band"]),
        "committed_hours": list(pilot_df["committed_hours_per_day"]),
        "results": results,
    }


@app.post("/api/stress_test")
def stress_test(req: StressTestRequest):
    """Evaluate the requested policies across four scarcity scenarios."""
    scenarios = [
        ("easy",     0.90, 0.02, 0.5),
        ("baseline", 0.70, 0.05, 0.4),
        ("hard",     0.50, 0.10, 0.4),
        ("extreme",  0.30, 0.15, 0.3),
    ]

    results = []
    for label, scarcity, sp, sf in scenarios:
        cfg = SimConfig(
            scarcity_factor=scarcity,
            shortage_prob=sp,
            shortage_factor=sf,
        )
        for policy_name in req.policies:
            sim = FeederSimulator(pilot_df, config=cfg)
            try:
                policy_fn = _resolve_policy(policy_name, sim)
            except HTTPException:
                continue
            stats = evaluate_policy(
                sim, policy_fn, n_episodes=req.episodes, base_seed=42
            )
            results.append({
                "scenario": label,
                "scarcity": scarcity,
                "policy": policy_name,
                "mean_reward": float(stats["mean_reward"]),
                "n_compliant": float(stats["compliance_rate"].sum()),
                "fairness": float(stats["fairness_index"]),
                "total_unserved_MWh": float(stats["mean_unserved_MWh"].sum()),
            })

    return {"results": results}
