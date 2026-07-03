# Adaptive Smart Grid Energy Allocation Using Reinforcement Learning

A reinforcement learning system that allocates electricity across IKEDC feeders
in Lagos, Nigeria to maximise regulatory band compliance under supply
constraints. Built on real NERC data (Q4 2022 quarterly report + MYTO 2022
service levels), evaluated against four baselines, and deployed as a React + FastAPI dashboard.

**Author:** LJ Nformi, BSc Software Engineering, African Leadership University
**Supervisor:** Emmanuel Adjei

> 🎥 **Demo video:** [[here](https://drive.google.com/drive/folders/1Tzh3p19AjEc-ms6CfIx3-Bixp18AQlZR?usp=sharing)]
> 🌐 **Live dashboard:** [[here](https://smartgrid-environemnt.vercel.app/)]
> 📄 **Thesis document:** [[here](https://docs.google.com/document/d/1UU_w5wqXIjnntBsiY9AVNBERVxgNWTtAQ0VPEJ79uM8/edit?usp=sharing)]

---

## Table of contents

1. [The problem](#the-problem)
2. [What this repository contains](#what-this-repository-contains)
3. [Quick start](#quick-start)
4. [Repository structure](#repository-structure)
5. [Full installation](#full-installation)
6. [Running the training pipeline](#running-the-training-pipeline)
7. [Running the dashboard](#running-the-dashboard)
8. [Reproducing reported results](#reproducing-reported-results)
9. [Testing strategy](#testing-strategy)
10. [Results summary](#results-summary)
11. [Analysis vs proposal objectives](#analysis-vs-proposal-objectives)
12. [Deployment](#deployment)
13. [Tech stack](#tech-stack)
14. [Known limitations and future work](#known-limitations-and-future-work)
15. [Acknowledgments](#acknowledgments)

---

## The problem

Ikeja Electric (IKEDC) is one of Nigeria's 11 electricity distribution
companies (DisCo's). It serves 414 feeders across Lagos, receiving roughly 1,113 GWh of
electricity per quarter, which is consistently less than total customer demand. The
Nigerian Electricity Regulatory Commission (NERC) groups feeders into five
service bands (A through E) with different daily hour commitments: Band A
feeders are entitled to at least 21 hours of supply per day, Band E to at
least 5. 

Existing approaches — priority-based rules, equal-split, classical
optimisation — either abandon low-priority bands under stress or fail to
adapt to demand and supply shocks. This project asks whether a
reinforcement learning agent trained on a simulator calibrated to real NERC
data can produce a single allocation policy that (a) matches or beats these
heuristics under normal conditions and (b) degrades gracefully when supply
tightens.

---

## What this repository contains

- **A data pipeline** that cleans and joins three real NERC datasets (MYTO 2022
  Appendix 2, monthly service caps for Oct-Dec 2022, and the Q4 2022 quarterly
  report) into a master feeder table with normalised demand weights.
- **A custom Gymnasium environment** simulating one-day episodes of hourly
  allocation across N feeders, calibrated against Q4 2022 supply totals.
- **A PPO training pipeline** using Stable-Baselines3, with two variants:
  fixed-scarcity training and domain-randomised training for robustness.
- **Four baseline policies** (equal split, priority-based,
  demand-proportional, deficit-chasing) used as evaluation references.
- **Multi-seed and stress-test infrastructure** for verifying reproducibility
  and off-distribution generalisation.
- **A React + FastAPI dashboard** that runs the trained models live against
  the same simulator used during training.

---

## Quick start

If you just want to see the dashboard running against the pre-trained models:

```bash
# 1. Clone
git clone https://github.com/<username>/mission-capstone.git
cd mission-capstone

# 2. Install Python deps
pip install -r requirements.txt

# 3. Start the backend (terminal 1)
cd dashboard/backend
uvicorn app:app --reload --port 8000

# 4. Install and start the frontend (terminal 2)
cd dashboard/frontend
npm install
npm run dev

# 5. Open http://localhost:5173
```

For full reproduction of the training results, see
[Reproducing reported results](#reproducing-reported-results).

---

## Repository structure

```
mission-capstone/
├── README.md                       this file
├── requirements.txt                Python dependencies
├── data/                           raw NERC datasets
│   ├── MYTO_2022_IKEDC_Appendix2_Service_Levels.csv
│   ├── Ikeja_Energy_Cap_October_2022.csv
│   ├── Ikeja_Energy_Cap_November_2022.csv
│   ├── Ikeja_Energy_Cap_December_2022.csv
│   ├── ikeja_quarterly_summary_2022_q3_vs_q4.csv
│   ├── ikeja_monthly_performance_jul_dec_2022.csv
│   └── ikeja_notes_and_context.txt      data provenance and known issues
├── output/                         generated artifacts (regenerable)
│   ├── master_feeder_table.csv
│   ├── pilot_feeders.csv
│   ├── simulation_parameters.json
│   ├── data_quality_report.txt
│   └── pilot_run/                  trained models and eval artifacts
│       ├── ppo_5feeder.zip
│       └── ppo_5feeder_robust.zip
├── scripts/                        pipeline scripts (run in order)
│   ├── build_tables.py             clean + join raw data
│   ├── select_pilot_feeders.py     pick 5 pilot feeders
│   ├── run_pilot.py                baseline evaluation
│   ├── train_ppo.py                fixed-scarcity PPO training
│   ├── train_ppo_robust.py         domain-randomised PPO training
│   ├── evaluate_ppo.py             load model, evaluate per-feeder
│   ├── train_multiseed.py          multi-seed reproducibility
│   └── stress_test.py              scarcity-scenario evaluation
├── feeder_env.py                   simulator + baselines + Gym wrapper
├── dashboard/                      React + FastAPI dashboard
│   ├── README.md
│   ├── backend/                    FastAPI, loads models at startup
│   └── frontend/                   Vite + React + Tailwind + Recharts
├── docs/
│   └── screenshots/                deployed dashboard screenshots
└── .gitignore
```

---

## Full installation

**Prerequisites**

- Python 3.10 or newer (3.12 tested)
- Node.js 18 or newer (for the dashboard frontend)
- Git

**Python dependencies**

```bash
pip install -r requirements.txt
```

`requirements.txt` contains:

```
numpy
pandas
matplotlib
gymnasium
stable-baselines3
torch
fastapi
uvicorn[standard]
```

If PyTorch installation fails (common on some Windows setups), install it
first following the platform-specific command at https://pytorch.org.

**Frontend dependencies**

```bash
cd dashboard/frontend
npm install
```

This downloads React, Vite, Tailwind, Recharts and lucide-react. Around 200 MB
of node_modules; takes 1-2 minutes on a decent connection.

---

## Running the training pipeline

The scripts are designed to be run in order. Each step consumes what the
previous one produced.

**Step 1: Build master tables**

Cleans and joins the three raw NERC datasets into a master feeder table with
normalised demand weights. Also produces supporting summary tables and a data
quality report.

```bash
python scripts/build_tables.py --input-dir ./data --output-dir ./output
```

Outputs:
- `output/master_feeder_table.csv` — 414 feeders, one row each
- `output/band_summary.csv`, `output/business_unit_summary.csv`
- `output/supply_budget.csv`, `output/historical_baselines.csv`
- `output/simulation_parameters.json`
- `output/data_quality_report.txt`

**Step 2: Select pilot feeders**

Picks one representative feeder per band (A, B, C, D, E) for the initial
experiments.

```bash
python scripts/select_pilot_feeders.py
```

**Step 3: Sanity-check with baselines**

Evaluates the four heuristic policies against the pilot feeders. Verifies the
simulator behaves sensibly before committing to PPO training.

```bash
python scripts/run_pilot.py --episodes 30
```

**Step 4: Train PPO**

Fixed-scarcity training at scarcity 0.7. Takes about 15-30 minutes on a laptop.

```bash
python scripts/train_ppo.py --timesteps 200000
```

**Step 5: Multi-seed reproducibility**

Trains PPO with 4 seeds to confirm results are not seed-dependent.

```bash
python scripts/train_multiseed.py --seeds 0 1 2 3 --timesteps 200000
```

**Step 6: Robust PPO training**

Trains PPO with domain randomisation over scarcity. Fixes the distribution-shift
problem revealed by the stress test.

```bash
python scripts/train_ppo_robust.py --timesteps 300000
```

**Step 7: Evaluate any model**

Loads a saved model and prints the per-feeder breakdown vs baselines.

```bash
python scripts/evaluate_ppo.py --model output/pilot_run/ppo_5feeder_robust.zip
```

**Step 8: Stress test**

Evaluates a trained model across four scarcity scenarios (easy, baseline, hard,
extreme).

```bash
python scripts/stress_test.py --model output/pilot_run/ppo_5feeder_robust.zip
```

---

## Running the dashboard

The dashboard runs the trained PPO models live against the same simulator used
during training. See `dashboard/README.md` for full setup details.

**Terminal 1 — Backend**

```bash
cd dashboard/backend
uvicorn app:app --reload --port 8000
```

The backend loads the PPO models from `output/pilot_run/` at startup and
exposes six REST endpoints. Auto-generated API documentation is available at
http://localhost:8000/docs.

**Terminal 2 — Frontend**

```bash
cd dashboard/frontend
npm run dev
```

Open http://localhost:5173. The dashboard has four pages:

1. **Overview** — Project context, headline metrics, pilot feeder cards
2. **Live Simulation** — Interactive 24-hour episode with per-feeder charts
3. **Compare Policies** — Multi-policy head-to-head over N episodes
4. **Stress Test** — Degradation curves across four scarcity levels

---

## Reproducing reported results

The numbers in the analysis section can be regenerated with the commands
below. Each returns roughly the reported value within noise.

| Reported result | Command | Expected output |
|---|---|---|
| Fixed PPO mean reward at scarcity 0.7 | `python scripts/train_multiseed.py --seeds 0 1 2 3` | ~4.47 ± 0.065 |
| Robust PPO mean reward at scarcity 0.7 | `python scripts/train_multiseed.py --seeds 0 1 2 3 --randomize-scarcity` | ~4.40 ± 0.100 |
| Compliance rate (PPO) | Included in above | 4.97-4.99 / 5 |
| Best baseline (demand-proportional) reward | `python scripts/run_pilot.py --episodes 100` | ~2.86 |
| Stress-test robust PPO at hard scarcity | `python scripts/stress_test.py --model output/pilot_run/ppo_5feeder_robust.zip` | ~3.89 compliant |

Sample outputs from a reference run are checked into
`output/pilot_run/sample_outputs/` for direct comparison.

---

## Testing strategy

The system was evaluated using **four testing strategies**:

**1. Baseline comparison across four heuristic policies.** Every PPO variant is
evaluated against four hand-coded baselines (equal split, priority-based,
demand-proportional, deficit-chasing) on the same simulator with the same
episode seeds. This gives a controlled reference point for what PPO's learned
policy is being compared against.

**2. Multi-seed reproducibility.** PPO is trained with 4 independent random
seeds (200k timesteps each for fixed-scarcity, 200k-300k for robust). Each
trained model is evaluated for 100 episodes. Across-seed standard deviation
quantifies training reproducibility; if the gap over the best baseline is
several times the across-seed std, the result is considered robust.

**3. Stress testing across four scarcity scenarios.** Each trained model is
evaluated at scarcity levels of 0.9 (easy), 0.7 (baseline / training
conditions), 0.5 (hard), and 0.3 (extreme). This distinguishes policies that
merely memorise the training distribution from policies that generalise.

**4. Varying input configurations via the dashboard.** The Compare and Stress
Test pages allow interactive evaluation with arbitrary scarcity, shortage
probability, episode counts, and policy selection. This tests the full API
surface against a range of inputs.

**Hardware / software environments tested**

| Environment | Purpose | Result |
|---|---|---|
| Windows 11 + Python 3.12 + PowerShell | Primary development and training | ✅ All scripts pass |
| Dashboard hosted deployment | Live evaluation on cloud infrastructure | ✅ See Deployment section |

**Testing evidence**

Screenshots and CSV outputs from representative runs are in `output/pilot_run/`
and `docs/screenshots/`. See also the video walkthrough linked at the top of
this document.

---

## Results summary

**Headline (multi-seed verified, 4 seeds, 100 evaluation episodes each):**

| Policy | Mean reward | Compliant feeders (of 5) | Fairness (Jain's) |
|---|---:|---:|---:|
| PPO (fixed scarcity) | **+4.47 ± 0.065** | **4.99** | 0.992 |
| PPO (robust) | +4.40 ± 0.100 | 4.97 | 0.994 |
| Demand-proportional | +2.86 | 4.47 | 0.934 |
| Equal split | -13.48 | 3.00 | 0.760 |
| Priority-based | -19.17 | 1.05 | 0.535 |
| Deficit-chasing | -30.18 | 0.00 | 0.558 |

The PPO advantage over the best baseline is **+1.60 with across-seed standard
deviation 0.065** — a signal-to-noise ratio of ~25:1, well above the 3-5
threshold typically used in RL methodology papers.

**Stress test — compliance across scarcity levels:**

| Scenario | Scarcity | Robust PPO | Demand-proportional | Fixed PPO |
|---|---:|---:|---:|---:|
| Easy | 0.90 | 5.00 | 5.00 | 5.00 |
| Baseline | 0.70 | 4.95 | 4.51 | 4.96 |
| Hard | 0.50 | **3.89** | 2.00 | 1.04 |
| Extreme | 0.30 | 0.75 | 0.04 | 0.05 |

Under harder conditions (scarcity 0.5), robust PPO holds nearly 4 of 5
feeders compliant while both the strongest baseline and the fixed-scarcity PPO
collapse. The cost of this robustness at training conditions is essentially
zero (4.47 → 4.40 reward, 4.99 → 4.97 compliance).

---

## Analysis vs proposal objectives

The original proposal committed to four objectives. This section documents
what was achieved against each and honestly notes what remains open.

**Objective 1: Formulate feeder-level power allocation as a reinforcement
learning problem grounded in NERC regulatory structure.** Achieved. The
proposal argued that IKEDC's allocation problem is best modelled as an MDP
with (a) supply as a stochastic exogenous input, (b) demand as a stochastic
per-feeder variable, (c) the action space as an allocation vector over
feeders, and (d) reward tied to NERC band compliance. The final simulator
implements exactly this structure. The state space is 17-dimensional; the
action space is 5-dimensional continuous (softmaxed to allocation shares);
the reward combines per-step energy penalties with end-of-episode compliance
and fairness bonuses.

**Objective 2: Train a PPO agent that outperforms baseline heuristics on
regulatory compliance.** Achieved and multi-seed verified. PPO scores +4.47
mean reward with 4.99/5 compliance against the best baseline's +2.86 and
4.47/5. The across-seed standard deviation of 0.065 confirms the result is
reproducible across independent training runs.

Note: the first iteration of the reward function was misspecified — it
weighted unserved energy by band priority, which double-counted the priority
signal already encoded in differential committed hours. PPO learned to
exploit this by abandoning low-priority bands. Diagnosing and fixing this
(via linear shortfall penalty and removal of the priority weight) was itself
a methodological contribution and is documented in the thesis discussion.

**Objective 3: Demonstrate robustness to changing supply conditions.**
Achieved. Domain randomisation over scarcity during training produced a
policy that maintains 3.89/5 compliance at scarcity 0.5 versus 2.00 for the
best baseline and 1.04 for a specialist PPO trained only at scarcity 0.7. The
in-distribution cost of this robustness is negligible (~0.07 reward).

**Objective 4: Build a working prototype for interactive demonstration.**
Achieved. The React + FastAPI dashboard loads the trained models at startup
and reuses the same simulator code as the training pipeline, ensuring that
values shown in the UI match those produced by the CLI scripts. Four pages
cover overview, live simulation, policy comparison and stress testing.

**Scope items deferred to future work**

- Scaling from the 5-feeder pilot to the full 414-feeder DisCo network was
  deliberately deferred to preserve methodological clarity. The proposal
  framed the pilot as a first step and the same scripts run on larger
  feeder sets without code changes.
- Validation against real SCADA telemetry data was outside the accessible
  scope — SCADA data for IKEDC is not publicly released. The current
  simulator calibrates against monthly totals from the NERC Q4 report but
  simulates hourly variation synthetically using a Lagos-typical load
  shape. This limitation is documented in the methodology.

---

## Deployment

**Deployed dashboard:** [[here](https://smartgrid-environemnt.vercel.app/)]

**Deployment plan.** The dashboard consists of two services with different
runtime requirements: a Python FastAPI backend that loads PyTorch-backed PPO
models (~1-2 MB each) and a static React frontend. The chosen approach:

- **Backend on Render.** Free tier is sufficient for demo
  traffic. The `Dockerfile` at `dashboard/backend/` pins Python 3.12,
  installs from `requirements.txt`, copies the trained models from `output/`,
  and starts uvicorn on the PORT environment variable. Cold start under 15
  seconds.
- **Frontend on Vercel.** The Vite build output is a directory of static
  assets served from a CDN. The `VITE_API_URL` build variable is set to the
  Render backend URL so production requests bypass the dev-only Vite proxy.

**Reproducing the deployment**

```bash
# Backend (Render)
cd dashboard/backend
# Push to a Git repository connected to Render, or use Render CLI
# Set start command: uvicorn app:app --host 0.0.0.0 --port $PORT

# Frontend (Vercel)
cd dashboard/frontend
# Push to a Git repository connected to Vercel
# Set build command: npm run build
# Set output directory: dist
# Set environment variable: VITE_API_URL=<your-backend-url>
```

**Verifying the deployment.** After both services are up:

1. Visit the deployed frontend URL.
2. The Overview page should load with 5 feeder cards.
3. Navigate to Live Simulation. Select PPO Robust, click Run. A chart of
   demand/supply/delivered curves and a per-feeder compliance panel should
   appear within 2-3 seconds.
4. Navigate to Stress Test. Click Run. Four scenario bars should populate
   within 30-60 seconds (100 episodes × 2 policies × 4 scenarios on
   free-tier CPU).

If any of the above fails, see the deployment logs and check that the
`VITE_API_URL` matches the actual backend URL.

---

## Tech stack

**Data pipeline**
- Python 3.12, pandas, numpy

**Simulator and training**
- Gymnasium (custom environment)
- Stable-Baselines3 (PPO implementation)
- PyTorch (SB3 backend)

**Dashboard backend**
- FastAPI (Python web framework)
- Uvicorn (ASGI server)
- Pydantic (request/response validation)

**Dashboard frontend**
- React 18 (UI library)
- Vite (build tool and dev server)
- Tailwind CSS (styling)
- Recharts (visualisation)
- lucide-react (icons)

**Deployment**
- Render / Railway (backend hosting)
- Vercel (frontend hosting)

---

## Known limitations and future work

**Simulator fidelity.** Hourly demand is synthetic (a Lagos-typical load shape
scaled to each feeder's monthly cap with noise). Monthly totals match the Q4
NERC report but individual hourly values are not validated against real
telemetry. A production system would require SCADA integration.

**Pilot scope.** Experiments are on a 5-feeder pilot rather than the full
414-feeder DisCo. Preliminary tests suggest the approach scales — the action
space grows linearly and PPO handles it — but full-scale training would
require substantially more compute time.

**Reward shape.** Compliance in the reward is a binary bonus above a
threshold; fairness uses Jain's index; neither is a NERC-published metric.
The current formulation is a reasonable proxy but a follow-up study should
compare against NERC's actual regulatory scoring once that becomes
programmatically accessible.

**Extreme scarcity.** At scarcity 0.3 (30% of demand can be met), all
policies degrade sharply. Some scarcity levels are infeasible — you cannot
serve 65 feeder-hours per day when only 35 are available. The system
degrades gracefully rather than catastrophically, but this ceiling exists
regardless of policy quality.

**Future directions**

- Multi-day episodes with energy-storage constraints
- Curriculum learning over scarcity rather than uniform randomisation
- Adversarial demand shocks during training
- Real-time deployment against SCADA feeds

---

## Acknowledgments

- Emmanuel Adjei (supervisor), for guidance throughout.
- African Leadership University, for the BSc Software Engineering programme
  under which this work was conducted.
- The Nigerian Electricity Regulatory Commission, for making the Q4 2022
  quarterly report and MYTO 2022 service-level standards publicly available.
- The Stable-Baselines3 team, whose implementation of PPO made experimental
  iteration tractable.

---

_See [[thesis document](https://docs.google.com/document/d/1UU_w5wqXIjnntBsiY9AVNBERVxgNWTtAQ0VPEJ79uM8/edit?usp=sharing)] for the full technical write-up._
