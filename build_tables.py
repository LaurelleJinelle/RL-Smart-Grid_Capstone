"""
Build a master feeder table and supporting tables from the IKEDC/NERC data
for the RL-based power-allocation project.

Inputs expected in --input-dir (defaults to ./data):
  - MYTO_2022_IKEDC_Appendix2_Service_Levels.csv
  - Ikeja_Energy_Cap_October_2022.csv
  - Ikeja_Energy_Cap_November_2022.csv
  - Ikeja_Energy_Cap_December_2022.csv
  - ikeja_quarterly_summary_2022_q3_vs_q4.csv
  - ikeja_monthly_performance_jul_dec_2022.csv

Outputs written to --output-dir (defaults to ./output):
  - master_feeder_table.csv        one row per feeder, fully joined
  - band_summary.csv               aggregates per tariff band
  - business_unit_summary.csv      aggregates per IKEDC business unit
  - supply_budget.csv              hourly / daily / monthly supply budgets
  - historical_baselines.csv       DisCo-level baselines for evaluation
  - simulation_parameters.json     calibration constants for the RL env
  - data_quality_report.txt        provenance, assumptions, mismatches

Run:
  python build_tables.py --input-dir ./data --output-dir ./output
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

# IKEDC band priority weights. Higher = more important.
# Used in the reward function and as a fallback ranking when caps are missing.
# Band A and "A - Bilateral" are treated as the same tier; bilateral is a
# contractual flag rather than a service-level distinction.
BAND_PRIORITY = {
    "A": 5,
    "A - Bilateral": 5,
    "B": 4,
    "C": 3,
    "D": 2,
    "E": 1,
}

# Minimum committed hours/day per band, per NERC Service Level standards.
# These are also encoded in the MYTO file as H:MM strings; we keep this
# dictionary as a fallback for feeders missing that column.
BAND_COMMITTED_HOURS = {
    "A": 20.0,
    "A - Bilateral": 20.0,
    "B": 16.0,
    "C": 12.0,
    "D": 8.0,
    "E": 4.0,
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def parse_duration_to_hours(value) -> float:
    """Convert an 'H:MM' or 'H:MM:SS' string into a decimal hours float.

    Returns np.nan if the value is missing or unparseable. We accept either
    H:MM (the format used by the 'Minimum Duration of Supply' column) or
    H:MM:SS (used by all the interruption / response columns).
    """
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if not s:
        return np.nan
    parts = s.split(":")
    try:
        parts = [int(p) for p in parts]
    except ValueError:
        return np.nan
    if len(parts) == 2:
        h, m = parts
        return h + m / 60.0
    if len(parts) == 3:
        h, m, sec = parts
        return h + m / 60.0 + sec / 3600.0
    return np.nan


def normalize_feeder_name(name) -> str:
    """Trim whitespace and collapse internal runs of spaces.

    Cross-file joins are by feeder name and the source CSVs are not always
    consistent about trailing spaces, so we normalize before joining.
    """
    if pd.isna(name):
        return ""
    return " ".join(str(name).split())


def load_caps_month(path: Path, month_label: str) -> pd.DataFrame:
    """Load one of the monthly caps files and aggregate by feeder.

    Some feeders appear under multiple business units in the same month with
    different cap values. We sum the cap across business units (treating
    those rows as different sections of the same feeder), and we keep the
    list of business units served as a separate column for traceability.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["Feeder Name"] = df["Feeder Name"].map(normalize_feeder_name)
    df = df.rename(
        columns={
            "Feeder Name": "feeder",
            "Non-MD Service Band": "band",
            "Cap (kWh)": "cap",
            "Business Unit": "business_unit",
        }
    )
    # Aggregate duplicate feeder names within the month.
    grouped = (
        df.groupby("feeder", as_index=False)
        .agg(
            cap=("cap", "sum"),
            band=("band", "first"),
            business_units=(
                "business_unit",
                lambda s: ";".join(sorted(set(s))),
            ),
            n_business_units=("business_unit", "nunique"),
        )
    )
    grouped = grouped.rename(
        columns={
            "cap": f"cap_{month_label}",
            "band": f"band_{month_label}",
            "business_units": f"business_units_{month_label}",
            "n_business_units": f"n_bus_{month_label}",
        }
    )
    return grouped


def load_myto(path: Path) -> pd.DataFrame:
    """Load the MYTO Appendix 2 service-level table and parse duration columns."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(
        columns={
            "Tariff Band": "band_myto",
            "Feeder": "feeder",
            "Minimum Duration of Supply (Hrs/Day)": "committed_hours_per_day",
            "Average Frequency of Interruptions Per Day": "avg_interruptions_per_day",
            "Average Duration of Interruptions": "avg_interruption_duration_str",
            "Average Response time to calls": "avg_response_time_calls_str",
            "Average Response time to resolving complaints": (
                "avg_response_time_complaints_str"
            ),
        }
    )
    df["feeder"] = df["feeder"].map(normalize_feeder_name)
    df["committed_hours_per_day"] = df["committed_hours_per_day"].map(
        parse_duration_to_hours
    )
    df["avg_interruption_duration_hours"] = df[
        "avg_interruption_duration_str"
    ].map(parse_duration_to_hours)
    df["avg_response_time_calls_hours"] = df["avg_response_time_calls_str"].map(
        parse_duration_to_hours
    )
    df["avg_response_time_complaints_hours"] = df[
        "avg_response_time_complaints_str"
    ].map(parse_duration_to_hours)
    df["avg_interruptions_per_day"] = pd.to_numeric(
        df["avg_interruptions_per_day"], errors="coerce"
    )
    keep = [
        "feeder",
        "band_myto",
        "committed_hours_per_day",
        "avg_interruptions_per_day",
        "avg_interruption_duration_hours",
        "avg_response_time_calls_hours",
        "avg_response_time_complaints_hours",
    ]
    return df[keep]


# ----------------------------------------------------------------------
# Core builders
# ----------------------------------------------------------------------

def build_master_feeder_table(
    myto: pd.DataFrame,
    caps_oct: pd.DataFrame,
    caps_nov: pd.DataFrame,
    caps_dec: pd.DataFrame,
) -> pd.DataFrame:
    """Outer-join MYTO with the three monthly caps and derive features."""
    df = myto.merge(caps_oct, on="feeder", how="outer")
    df = df.merge(caps_nov, on="feeder", how="outer")
    df = df.merge(caps_dec, on="feeder", how="outer")

    # Provenance flags so downstream code can tell where each row came from.
    df["in_myto"] = df["band_myto"].notna()
    df["in_caps_oct"] = df["cap_oct"].notna()
    df["in_caps_nov"] = df["cap_nov"].notna()
    df["in_caps_dec"] = df["cap_dec"].notna()
    df["months_present"] = (
        df["in_caps_oct"].astype(int)
        + df["in_caps_nov"].astype(int)
        + df["in_caps_dec"].astype(int)
    )

    # Resolve a single canonical band per feeder. Bands don't change month
    # to month in the caps data (verified empirically: 0 changes across 279
    # feeders present in all three months), so we just take the first
    # non-null band in MYTO -> Oct -> Nov -> Dec order.
    df["band"] = (
        df["band_myto"]
        .fillna(df["band_oct"])
        .fillna(df["band_nov"])
        .fillna(df["band_dec"])
    )

    # Average cap across whichever months are available. Feeders with no cap
    # at all in any month will get np.nan here and a fallback weight later.
    cap_cols = ["cap_oct", "cap_nov", "cap_dec"]
    df["cap_mean"] = df[cap_cols].mean(axis=1)
    df["cap_total_q4"] = df[cap_cols].sum(axis=1, min_count=1)

    # Carry through a primary business unit (use Oct's; falls back to Nov,
    # then Dec). Multiple BUs are preserved in the *_business_units_* cols.
    df["primary_business_units"] = (
        df["business_units_oct"]
        .fillna(df["business_units_nov"])
        .fillna(df["business_units_dec"])
    )

    # Priority weight and committed hours, with fallbacks for missing data.
    df["band_priority"] = df["band"].map(BAND_PRIORITY)
    df["committed_hours_per_day"] = df["committed_hours_per_day"].fillna(
        df["band"].map(BAND_COMMITTED_HOURS)
    )

    # Demand weight: relative share of total demand each feeder represents.
    # Caps are the strongest signal we have; for feeders without a cap we
    # impute the mean cap *within their band* (which preserves the band's
    # average size), and for feeders without any band (vanishingly rare) we
    # impute the global mean cap.
    band_mean_cap = df.groupby("band")["cap_mean"].transform("mean")
    df["cap_imputed_flag"] = df["cap_mean"].isna()
    df["cap_for_weight"] = df["cap_mean"].fillna(band_mean_cap)
    df["cap_for_weight"] = df["cap_for_weight"].fillna(df["cap_mean"].mean())

    # Normalize so weights sum to 1 across the master table.
    total = df["cap_for_weight"].sum()
    df["demand_weight"] = df["cap_for_weight"] / total

    # Order columns sensibly.
    leading = [
        "feeder",
        "band",
        "band_priority",
        "committed_hours_per_day",
        "primary_business_units",
        "cap_oct",
        "cap_nov",
        "cap_dec",
        "cap_mean",
        "cap_total_q4",
        "cap_for_weight",
        "cap_imputed_flag",
        "demand_weight",
        "avg_interruptions_per_day",
        "avg_interruption_duration_hours",
        "avg_response_time_calls_hours",
        "avg_response_time_complaints_hours",
        "in_myto",
        "in_caps_oct",
        "in_caps_nov",
        "in_caps_dec",
        "months_present",
    ]
    trailing = [c for c in df.columns if c not in leading]
    df = df[leading + trailing]
    return df.sort_values(["band_priority", "feeder"], ascending=[False, True]).reset_index(drop=True)


def build_band_summary(master: pd.DataFrame) -> pd.DataFrame:
    """Per-band aggregates for environment configuration and validation."""
    grp = master.groupby("band", dropna=False)
    summary = grp.agg(
        n_feeders=("feeder", "count"),
        band_priority=("band_priority", "first"),
        committed_hours_per_day=("committed_hours_per_day", "mean"),
        mean_cap=("cap_mean", "mean"),
        median_cap=("cap_mean", "median"),
        total_cap_q4=("cap_total_q4", "sum"),
        sum_demand_weight=("demand_weight", "sum"),
        mean_interruptions_per_day=("avg_interruptions_per_day", "mean"),
        mean_interruption_duration_hours=(
            "avg_interruption_duration_hours",
            "mean",
        ),
    ).reset_index()
    return summary.sort_values("band_priority", ascending=False).reset_index(drop=True)


def build_business_unit_summary(master: pd.DataFrame) -> pd.DataFrame:
    """Aggregate by business unit.

    Some feeders span multiple BUs; we expand the semicolon-separated
    primary_business_units field so a feeder serving two BUs contributes
    its weight (split evenly) to each.
    """
    rows = []
    for _, r in master.iterrows():
        if pd.isna(r["primary_business_units"]):
            continue
        bus = r["primary_business_units"].split(";")
        share = 1.0 / len(bus)
        for bu in bus:
            rows.append(
                {
                    "business_unit": bu.strip(),
                    "feeder": r["feeder"],
                    "band": r["band"],
                    "band_priority": r["band_priority"],
                    "cap_share": (r["cap_mean"] or 0.0) * share if pd.notna(r["cap_mean"]) else 0.0,
                    "weight_share": r["demand_weight"] * share,
                }
            )
    exp = pd.DataFrame(rows)
    if exp.empty:
        return exp
    return (
        exp.groupby("business_unit")
        .agg(
            n_feeder_slots=("feeder", "count"),
            mean_band_priority=("band_priority", "mean"),
            total_cap=("cap_share", "sum"),
            total_weight=("weight_share", "sum"),
        )
        .reset_index()
        .sort_values("total_weight", ascending=False)
    )


def build_supply_budget(monthly_perf: pd.DataFrame, quarterly: pd.DataFrame) -> pd.DataFrame:
    """Build hourly / daily / monthly supply and demand totals for Q4 2022.

    We use the monthly Energy_Received_GWh column for supply and
    Energy_Billed_GWh for delivered demand. The hourly average is the
    monthly total divided by (days_in_month * 24). The proposal treats
    hourly demand as synthetic but calibrated to these totals.
    """
    # Restrict to Q4 (Oct-Dec).
    q4 = monthly_perf[monthly_perf["Month"].isin(["2022-10", "2022-11", "2022-12"])].copy()
    days_in_month = {"2022-10": 31, "2022-11": 30, "2022-12": 31}
    q4["days_in_month"] = q4["Month"].map(days_in_month)
    q4["hours_in_month"] = q4["days_in_month"] * 24
    q4["supply_received_MWh"] = q4["Energy_Received_GWh"] * 1000.0
    q4["demand_billed_MWh"] = q4["Energy_Billed_GWh"] * 1000.0
    q4["supply_avg_MWh_per_hour"] = q4["supply_received_MWh"] / q4["hours_in_month"]
    q4["demand_avg_MWh_per_hour"] = q4["demand_billed_MWh"] / q4["hours_in_month"]
    q4["billing_efficiency"] = q4["Energy_Billed_GWh"] / q4["Energy_Received_GWh"]
    keep = [
        "Month",
        "days_in_month",
        "hours_in_month",
        "supply_received_MWh",
        "demand_billed_MWh",
        "supply_avg_MWh_per_hour",
        "demand_avg_MWh_per_hour",
        "billing_efficiency",
    ]
    return q4[keep].reset_index(drop=True)


def build_historical_baselines(quarterly: pd.DataFrame) -> pd.DataFrame:
    """Filter the quarterly summary into the rows most relevant for evaluation."""
    keys = [
        "Average hourly energy received at trading points",
        "Available Partial Contracted Capacity (PCC)",
        "Energy offtake variance vs available PCC",
        "Energy received",
        "Energy billed",
        "Billing efficiency",
        "Billings",
        "Collections",
        "Collection efficiency",
        "ATC&C loss",
        "Customers (registered)",
        "Customers (metered)",
        "Metering rate",
        "Complaints received",
        "Complaints resolved",
        "Complaint resolution rate",
    ]
    mask = quarterly["Metric"].isin(keys)
    return quarterly[mask].reset_index(drop=True)


def build_simulation_parameters(
    master: pd.DataFrame,
    supply_budget: pd.DataFrame,
    quarterly: pd.DataFrame,
) -> dict:
    """Bundle the constants the RL environment needs into a single JSON file."""
    # Pull Q4 totals from the quarterly summary as a sanity anchor.
    def _q4(metric):
        row = quarterly[quarterly["Metric"] == metric]
        return float(row["2022_Q4"].iloc[0]) if not row.empty else None

    params = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "quarter": "2022_Q4",
        "n_feeders": int(len(master)),
        "n_feeders_with_caps": int(master["cap_mean"].notna().sum()),
        "n_feeders_imputed": int(master["cap_imputed_flag"].sum()),
        "bands": sorted(master["band"].dropna().unique().tolist()),
        "band_priority": BAND_PRIORITY,
        "band_committed_hours_default": BAND_COMMITTED_HOURS,
        "supply": {
            "q4_total_energy_received_GWh": _q4("Energy received"),
            "q4_total_energy_billed_GWh": _q4("Energy billed"),
            "q4_billing_efficiency_pct": _q4("Billing efficiency"),
            "q4_atcc_loss_pct": _q4("ATC&C loss"),
            "avg_MWh_per_hour_q4": float(
                supply_budget["supply_avg_MWh_per_hour"].mean()
            ),
            "monthly_supply_MWh": dict(
                zip(
                    supply_budget["Month"].tolist(),
                    supply_budget["supply_received_MWh"].round(2).tolist(),
                )
            ),
        },
        "episode_defaults": {
            "hours_per_step": 1,
            "steps_per_episode": 24,
            "episodes_per_quarter": 90,
        },
        "reward_weights_starting_point": {
            "unserved_energy_penalty": 1.0,
            "band_compliance_bonus": 0.5,
            "fairness_bonus": 0.3,
            "wasted_energy_penalty": 0.2,
        },
        "notes": [
            "demand_weight in master_feeder_table sums to 1.0 across feeders.",
            "Multiply demand_weight by supply.avg_MWh_per_hour_q4 (or a "
            "scaled-up demand total) to get a per-feeder hourly demand "
            "anchor before adding load shape and noise.",
            "Cap units in the source CSV are labelled 'kWh' but the summed "
            "monthly totals are consistent with MWh; the script preserves "
            "the original numbers without changing units.",
        ],
    }
    return params


# ----------------------------------------------------------------------
# Data quality report
# ----------------------------------------------------------------------

def write_data_quality_report(
    master: pd.DataFrame,
    myto: pd.DataFrame,
    caps_oct: pd.DataFrame,
    caps_nov: pd.DataFrame,
    caps_dec: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = []
    add = lines.append
    add("Data quality report")
    add("=" * 60)
    add(f"Generated at: {datetime.now(timezone.utc).isoformat()}")
    add("")
    add("Row counts")
    add("-" * 60)
    add(f"  MYTO Appendix 2:           {len(myto):>4} feeders")
    add(f"  Caps October (post-agg):   {len(caps_oct):>4} feeders")
    add(f"  Caps November (post-agg):  {len(caps_nov):>4} feeders")
    add(f"  Caps December (post-agg):  {len(caps_dec):>4} feeders")
    add(f"  Master feeder table:       {len(master):>4} feeders (outer join)")
    add("")

    add("Coverage flags")
    add("-" * 60)
    add(f"  In MYTO + all three months: "
        f"{int(((master['in_myto']) & (master['months_present'] == 3)).sum())}")
    add(f"  In MYTO only (no caps):     "
        f"{int(((master['in_myto']) & (master['months_present'] == 0)).sum())}")
    add(f"  In caps only (no MYTO):     "
        f"{int(((~master['in_myto']) & (master['months_present'] > 0)).sum())}")
    add(f"  Caps imputed from band mean: {int(master['cap_imputed_flag'].sum())}")
    add("")

    add("Band distribution in master table")
    add("-" * 60)
    band_counts = master["band"].value_counts(dropna=False).sort_index()
    for b, n in band_counts.items():
        add(f"  {str(b):<16} {n:>4}")
    add("")

    add("Demand weight sanity checks")
    add("-" * 60)
    add(f"  Sum of demand_weight: {master['demand_weight'].sum():.6f}  (should be 1.0)")
    add(f"  Min demand_weight:    {master['demand_weight'].min():.6f}")
    add(f"  Max demand_weight:    {master['demand_weight'].max():.6f}")
    add(f"  Median demand_weight: {master['demand_weight'].median():.6f}")
    add("")

    add("Cap totals (units as in source CSV; interpretable as MWh/month)")
    add("-" * 60)
    add(f"  October cap sum:  {master['cap_oct'].sum():>12,.0f}")
    add(f"  November cap sum: {master['cap_nov'].sum():>12,.0f}")
    add(f"  December cap sum: {master['cap_dec'].sum():>12,.0f}")
    add("")

    add("Assumptions to flag in the methodology section")
    add("-" * 60)
    add("  1. Cap units: source column is labelled 'Cap (kWh)' but the "
        "monthly totals are consistent with MWh. The script keeps the "
        "original values unchanged.")
    add("  2. Some feeders appear under multiple business units in the "
        "caps files; their caps are summed across BUs to produce one "
        "row per feeder per month.")
    add("  3. Feeders present in MYTO but absent from a caps file get "
        "their cap imputed from the within-band mean.")
    add("  4. Bands are stable across Oct/Nov/Dec (empirically verified, "
        "0 of 279 common feeders change band).")
    add("  5. 'A - Bilateral' is treated as priority-equivalent to Band A.")
    add("")

    output_path.write_text("\n".join(lines))


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="./data", type=Path)
    parser.add_argument("--output-dir", default="./output", type=Path)
    args = parser.parse_args(argv)

    in_dir = args.input_dir
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading inputs from: {in_dir.resolve()}")
    print(f"Writing outputs to:  {out_dir.resolve()}")
    print()

    # 1. Load.
    myto = load_myto(in_dir / "MYTO_2022_IKEDC_Appendix2_Service_Levels.csv")
    caps_oct = load_caps_month(in_dir / "Ikeja_Energy_Cap_October_2022.csv", "oct")
    caps_nov = load_caps_month(in_dir / "Ikeja_Energy_Cap_November_2022.csv", "nov")
    caps_dec = load_caps_month(in_dir / "Ikeja_Energy_Cap_December_2022.csv", "dec")
    quarterly = pd.read_csv(in_dir / "ikeja_quarterly_summary_2022_q3_vs_q4.csv")
    monthly_perf = pd.read_csv(in_dir / "ikeja_monthly_performance_jul_dec_2022.csv")

    print(f"  MYTO:               {len(myto)} feeders")
    print(f"  Caps October:       {len(caps_oct)} feeders (post-aggregation)")
    print(f"  Caps November:      {len(caps_nov)} feeders (post-aggregation)")
    print(f"  Caps December:      {len(caps_dec)} feeders (post-aggregation)")
    print()

    # 2. Build the master feeder table and supporting tables.
    master = build_master_feeder_table(myto, caps_oct, caps_nov, caps_dec)
    band_summary = build_band_summary(master)
    bu_summary = build_business_unit_summary(master)
    supply_budget = build_supply_budget(monthly_perf, quarterly)
    baselines = build_historical_baselines(quarterly)
    sim_params = build_simulation_parameters(master, supply_budget, quarterly)

    # 3. Write all outputs.
    master.to_csv(out_dir / "master_feeder_table.csv", index=False)
    band_summary.to_csv(out_dir / "band_summary.csv", index=False)
    bu_summary.to_csv(out_dir / "business_unit_summary.csv", index=False)
    supply_budget.to_csv(out_dir / "supply_budget.csv", index=False)
    baselines.to_csv(out_dir / "historical_baselines.csv", index=False)
    (out_dir / "simulation_parameters.json").write_text(
        json.dumps(sim_params, indent=2)
    )
    write_data_quality_report(
        master, myto, caps_oct, caps_nov, caps_dec,
        out_dir / "data_quality_report.txt",
    )

    print("Outputs written:")
    for name in [
        "master_feeder_table.csv",
        "band_summary.csv",
        "business_unit_summary.csv",
        "supply_budget.csv",
        "historical_baselines.csv",
        "simulation_parameters.json",
        "data_quality_report.txt",
    ]:
        path = out_dir / name
        print(f"  {name:<32} {path.stat().st_size:>8} bytes")
    print()

    # 4. Sanity assertions.
    assert abs(master["demand_weight"].sum() - 1.0) < 1e-9, "demand weights must sum to 1"
    assert master["feeder"].is_unique, "feeder names must be unique in master"
    assert master["band"].notna().all(), "every feeder must have a band"
    print(f"All sanity checks passed. Master feeder table has {len(master)} rows.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
