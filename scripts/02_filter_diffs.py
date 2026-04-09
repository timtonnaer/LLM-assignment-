"""
Script 02: Filter Diffs
Finds all diff JSON files for the selected firms within the 2015-2023 window
and builds an index CSV.

Usage:
  python scripts/02_filter_diffs.py               # pilot (default)
  python scripts/02_filter_diffs.py --sample full # all ~460 S&P 500 firms
"""

import json
import argparse
import pandas as pd
from pathlib import Path

OUTPUTS_DIR = Path("/Users/timtonnaer/risk_project/outputs")
DIFFS_DIR   = Path("/Users/timtonnaer/risk_project/outputs/diffs")

FIRMS_CSV_MAP = {
    "pilot": OUTPUTS_DIR / "pilot_firms.csv",
    "full":  OUTPUTS_DIR / "all_firms.csv",
}
INDEX_CSV_MAP = {
    "pilot": OUTPUTS_DIR / "pilot_diff_index.csv",
    "full":  OUTPUTS_DIR / "all_diff_index.csv",
}

YEAR_NEW_MIN = 2016
YEAR_NEW_MAX = 2023


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", choices=["pilot", "full"], default="pilot",
                        help="Which firm list to use (default: pilot)")
    args = parser.parse_args()

    firms_csv  = FIRMS_CSV_MAP[args.sample]
    output_csv = INDEX_CSV_MAP[args.sample]

    firms = pd.read_csv(firms_csv)
    firm_ciks = set(firms["cik"].astype(str))
    n_firms = len(firm_ciks)
    print(f"Firms loaded: {n_firms} ({args.sample} sample)")

    records = []
    missing_by_firm = {}

    for cik in firm_ciks:
        row = firms.loc[firms["cik"].astype(str) == cik].iloc[0]
        ticker = row["ticker"]
        found_years = []

        for year_new in range(YEAR_NEW_MIN, YEAR_NEW_MAX + 1):
            year_old = year_new - 1
            fpath = DIFFS_DIR / f"{cik}_{year_old}_{year_new}.json"

            if fpath.exists():
                with open(fpath) as f:
                    d = json.load(f)
                records.append({
                    "cik":        cik,
                    "ticker":     ticker,
                    "year_old":   year_old,
                    "year_new":   year_new,
                    "filepath":   str(fpath),
                    "n_added":    d.get("n_added", 0),
                    "n_removed":  d.get("n_removed", 0),
                    "similarity": d.get("similarity", None),
                    "added_ratio":d.get("added_ratio", None),
                })
                found_years.append(year_new)

        missing = [y for y in range(YEAR_NEW_MIN, YEAR_NEW_MAX + 1) if y not in found_years]
        if missing:
            missing_by_firm[ticker] = missing

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)

    expected = n_firms * (YEAR_NEW_MAX - YEAR_NEW_MIN + 1)
    print(f"\nTotal diff files found : {len(df)}")
    print(f"Expected ({n_firms} firms × 8 years): {expected}")
    print(f"Coverage: {len(df)/expected*100:.1f}%")
    print(f"Firms with complete coverage: {n_firms - len(missing_by_firm)}/{n_firms}")

    if missing_by_firm:
        print(f"\nFirms with gaps ({len(missing_by_firm)}) — sample:")
        for ticker, years in list(sorted(missing_by_firm.items()))[:10]:
            print(f"  {ticker}: missing year_new {years}")
        if len(missing_by_firm) > 10:
            print(f"  ... and {len(missing_by_firm)-10} more")

    print(f"\nSaved index to {output_csv}")
    print(f"\nSummary stats:")
    print(df[["n_added", "n_removed", "similarity"]].describe().round(2))


if __name__ == "__main__":
    main()
