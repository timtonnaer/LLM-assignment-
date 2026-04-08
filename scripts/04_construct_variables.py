"""
Script 04: Construct Firm-Year Variables
Aggregates LLM classification results to a firm-year panel with key research variables.
Merges in diff-level statistics (similarity, n_added) and firm metadata.
"""

import json
import pandas as pd
from pathlib import Path

DIFF_INDEX_CSV = Path("/Users/timtonnaer/risk_project/outputs/pilot_diff_index.csv")
PILOT_FIRMS_CSV = Path("/Users/timtonnaer/risk_project/outputs/pilot_firms.csv")
CLASSIFIED_DIR = Path("/Users/timtonnaer/risk_project/outputs/classified")
OUTPUT_CSV = Path("/Users/timtonnaer/risk_project/outputs/firm_year_variables.csv")

RISK_TYPES = ["operational", "cyber", "regulatory", "supply_chain", "financing", "macroeconomic", "other"]


def load_classified(cik: str, year_old: int, year_new: int) -> list[dict]:
    path = CLASSIFIED_DIR / f"{cik}_{year_old}_{year_new}_classified.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def compute_variables(classifications: list[dict], diff_row: pd.Series) -> dict:
    n_added_raw = diff_row["n_added"]

    if not classifications:
        base = {
            "n_classified": 0,
            "risk_update_intensity": 0,
            "boilerplate_ratio": None,
            "vagueness_ratio": None,
            "n_added_total": n_added_raw,
            "similarity": diff_row["similarity"],
            "added_ratio": diff_row["added_ratio"],
        }
        for rt in RISK_TYPES:
            base[f"risk_{rt}"] = 0
        return base

    n_total = len(classifications)

    # Nature counts
    n_meaningful = sum(1 for c in classifications if c.get("nature") in {"new_risk", "expanded_existing"})
    n_boilerplate = sum(1 for c in classifications if c.get("nature") == "boilerplate")

    # Style counts (exclude unknowns)
    style_known = [c for c in classifications if c.get("style") in {"concrete", "vague"}]
    n_vague = sum(1 for c in style_known if c.get("style") == "vague")

    # Risk type counts
    risk_counts = {rt: sum(1 for c in classifications if c.get("risk_type") == rt) for rt in RISK_TYPES}

    return {
        "n_classified": n_total,
        "risk_update_intensity": n_meaningful,
        "boilerplate_ratio": n_boilerplate / n_total if n_total > 0 else None,
        "vagueness_ratio": n_vague / len(style_known) if style_known else None,
        "n_added_total": n_added_raw,
        "similarity": diff_row["similarity"],
        "added_ratio": diff_row["added_ratio"],
        **{f"risk_{rt}": risk_counts[rt] for rt in RISK_TYPES},
    }


def main():
    index = pd.read_csv(DIFF_INDEX_CSV)
    firms = pd.read_csv(PILOT_FIRMS_CSV)
    firms["cik"] = firms["cik"].astype(str)

    records = []
    missing_classifications = []

    for _, row in index.iterrows():
        cik = str(row["cik"])
        year_old = int(row["year_old"])
        year_new = int(row["year_new"])

        classifications = load_classified(cik, year_old, year_new)
        if not classifications:
            missing_classifications.append(f"{row['ticker']} {year_new}")

        variables = compute_variables(classifications, row)
        records.append({
            "cik": cik,
            "ticker": row["ticker"],
            "year_old": year_old,
            "year_new": year_new,
            **variables,
        })

    df = pd.DataFrame(records)

    # Merge sector
    df = df.merge(firms[["cik", "sector", "company_name"]], on="cik", how="left")

    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Firm-year panel: {len(df)} rows")
    print(f"Firms covered: {df['ticker'].nunique()}")
    print(f"Years covered: {sorted(df['year_new'].unique())}")

    classified_mask = df["n_classified"] > 0
    print(f"Rows with LLM classifications: {classified_mask.sum()} / {len(df)}")

    if missing_classifications:
        print(f"\nRows missing classifications ({len(missing_classifications)}) — run script 03 first:")
        for m in missing_classifications[:10]:
            print(f"  {m}")
        if len(missing_classifications) > 10:
            print(f"  ... and {len(missing_classifications) - 10} more")

    print(f"\nKey variable summary (classified rows only):")
    cols = ["risk_update_intensity", "boilerplate_ratio", "vagueness_ratio", "similarity",
            "risk_cyber", "risk_operational", "risk_regulatory", "risk_financing"]
    print(df.loc[classified_mask, cols].describe().round(3).to_string())
    print(f"\nSaved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
