"""
Script 01: Extract Pilot Firms
Reads the 50 pilot firms from the Excel file and saves them to a CSV.
"""

import pandas as pd
from pathlib import Path

EXCEL_PATH = Path("/Users/timtonnaer/Downloads/SP500_Research_Firmlist.xlsx")
OUTPUT_PATH = Path("/Users/timtonnaer/risk_project/outputs/pilot_firms.csv")


def main():
    df = pd.read_excel(EXCEL_PATH, sheet_name="Pilot Sample (50 firms)")
    print(f"Columns found: {df.columns.tolist()}")
    print(f"Shape: {df.shape}")

    # Normalize column names
    df.columns = df.columns.str.strip()

    # Identify and rename key columns robustly
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

    # Keep only needed columns (handle missing gracefully)
    keep = [c for c in ["ticker", "company_name", "sector", "cik"] if c in df.columns]
    df = df[keep].dropna(subset=["cik"])

    # Ensure CIK is integer string (no decimals)
    df["cik"] = df["cik"].astype(float).astype(int).astype(str)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} firms to {OUTPUT_PATH}")
    print(df.to_string())


if __name__ == "__main__":
    main()
