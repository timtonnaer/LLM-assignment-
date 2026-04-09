"""
Script 01: Extract Firms
Reads firms from the Excel file and saves them to a CSV.

Usage:
  python scripts/01_extract_pilot_firms.py               # pilot 50 firms (default)
  python scripts/01_extract_pilot_firms.py --sample full # all ~460 S&P 500 firms
"""

import argparse
import pandas as pd
from pathlib import Path

EXCEL_PATH   = Path("/Users/timtonnaer/Downloads/SP500_Research_Firmlist.xlsx")
OUTPUTS_DIR  = Path("/Users/timtonnaer/risk_project/outputs")

SHEET_MAP = {
    "pilot": "Pilot Sample (50 firms)",
    "full":  "Full Sample (~500 firms)",
}
OUTPUT_MAP = {
    "pilot": OUTPUTS_DIR / "pilot_firms.csv",
    "full":  OUTPUTS_DIR / "all_firms.csv",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", choices=["pilot", "full"], default="pilot",
                        help="Which sample to extract (default: pilot)")
    args = parser.parse_args()

    sheet  = SHEET_MAP[args.sample]
    output = OUTPUT_MAP[args.sample]

    df = pd.read_excel(EXCEL_PATH, sheet_name=sheet)
    print(f"Sheet: '{sheet}' | Rows: {len(df)} | Columns: {df.columns.tolist()}")

    df.columns = df.columns.str.strip()

    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if "ticker" in cl:
            col_map[col] = "ticker"
        elif "company" in cl or "name" in cl:
            col_map[col] = "company_name"
        elif "gics" in cl or "sector" in cl:
            col_map[col] = "sector"
        elif "cik" in cl:
            col_map[col] = "cik"

    df = df.rename(columns=col_map)
    keep = [c for c in ["ticker", "company_name", "sector", "cik"] if c in df.columns]
    df = df[keep].dropna(subset=["cik"])
    df["cik"] = df["cik"].astype(float).astype(int).astype(str)

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} firms to {output}")

    # Sector breakdown
    if "sector" in df.columns:
        print("\nFirms by sector:")
        print(df["sector"].value_counts().to_string())


if __name__ == "__main__":
    main()
