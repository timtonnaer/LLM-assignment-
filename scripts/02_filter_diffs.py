"""
Script 02: Filter Diffs
Finds all diff JSON files for the 50 pilot firms within the 2015-2023 window
and builds an index CSV.
"""

import json
import pandas as pd
from pathlib import Path

PILOT_FIRMS_CSV = Path("/Users/timtonnaer/risk_project/outputs/pilot_firms.csv")
DIFFS_DIR = Path("/Users/timtonnaer/risk_project/outputs/diffs")
OUTPUT_CSV = Path("/Users/timtonnaer/risk_project/outputs/pilot_diff_index.csv")

# year_new in 2016..2023 covers t-1 vs t comparisons for the 2015-2023 window
YEAR_NEW_MIN = 2016
YEAR_NEW_MAX = 2023


def main():
    firms = pd.read_csv(PILOT_FIRMS_CSV)
    pilot_ciks = set(firms["cik"].astype(str))
    print(f"Pilot CIKs loaded: {len(pilot_ciks)}")

    records = []
    missing_by_firm = {}

    for cik in pilot_ciks:
        ticker = firms.loc[firms["cik"].astype(str) == cik, "ticker"].iloc[0]
        found_years = []

        for year_new in range(YEAR_NEW_MIN, YEAR_NEW_MAX + 1):
            year_old = year_new - 1
            fname = f"{cik}_{year_old}_{year_new}.json"
            fpath = DIFFS_DIR / fname

            if fpath.exists():
                with open(fpath) as f:
                    d = json.load(f)
                records.append({
                    "cik": cik,
                    "ticker": ticker,
                    "year_old": year_old,
                    "year_new": year_new,
                    "filepath": str(fpath),
                    "n_added": d.get("n_added", 0),
                    "n_removed": d.get("n_removed", 0),
                    "similarity": d.get("similarity", None),
                    "added_ratio": d.get("added_ratio", None),
                })
                found_years.append(year_new)

        expected = list(range(YEAR_NEW_MIN, YEAR_NEW_MAX + 1))
        missing = [y for y in expected if y not in found_years]
        if missing:
            missing_by_firm[ticker] = missing

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nTotal diff files found: {len(df)}")
    print(f"Expected (50 firms × 8 years): 400")
    print(f"Coverage: {len(df)/400*100:.1f}%")

    if missing_by_firm:
        print(f"\nFirms with missing year pairs ({len(missing_by_firm)}):")
        for ticker, years in sorted(missing_by_firm.items()):
            print(f"  {ticker}: missing year_new {years}")
    else:
        print("\nAll firms have complete coverage.")

    print(f"\nSaved index to {OUTPUT_CSV}")
    print(f"\nSummary stats:")
    print(df[["n_added", "n_removed", "similarity"]].describe().round(2))


if __name__ == "__main__":
    main()
