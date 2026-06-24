"""
Pick 5 representative feeders from the master feeder table, one per band
(A, B, C, D, E). For each band we want a feeder that:
  - has cap data in all three months (no imputation),
  - is closest to the median cap within its band (a "typical" feeder),
  - has real MYTO interruption stats.

Skips "A - Bilateral" because it's a contractual variant of Band A rather
than a separate service tier.

Run:
  python select_pilot_feeders.py --master ./output/master_feeder_table.csv --out ./output/pilot_feeders.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BANDS = ["A", "B", "C", "D", "E"]


def pick_median_feeder(df: pd.DataFrame) -> pd.Series:
    """From a band-filtered dataframe, return the row closest to median cap."""
    median = df["cap_mean"].median()
    idx = (df["cap_mean"] - median).abs().idxmin()
    return df.loc[idx]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", default="./output/master_feeder_table.csv", type=Path)
    parser.add_argument("--out", default="./output/pilot_feeders.csv", type=Path)
    args = parser.parse_args()

    master = pd.read_csv(args.master)

    picks = []
    for band in BANDS:
        # Keep only feeders that are clean: in MYTO, in all 3 months, no imputation.
        eligible = master[
            (master["band"] == band)
            & (master["in_myto"])
            & (master["in_caps_oct"])
            & (master["in_caps_nov"])
            & (master["in_caps_dec"])
            & (~master["cap_imputed_flag"])
            & (master["avg_interruptions_per_day"].notna())
        ].copy()
        if eligible.empty:
            print(f"WARNING: no clean feeder for band {band}; relaxing constraints")
            eligible = master[master["band"] == band].copy()
        picks.append(pick_median_feeder(eligible))

    pilot = pd.DataFrame(picks).reset_index(drop=True)
    pilot = pilot[
        [
            "feeder",
            "band",
            "band_priority",
            "committed_hours_per_day",
            "primary_business_units",
            "cap_oct",
            "cap_nov",
            "cap_dec",
            "cap_mean",
            "demand_weight",
            "avg_interruptions_per_day",
            "avg_interruption_duration_hours",
        ]
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pilot.to_csv(args.out, index=False)

    print(f"Picked {len(pilot)} pilot feeders, written to {args.out}")
    print()
    print(pilot.to_string(index=False))


if __name__ == "__main__":
    main()
