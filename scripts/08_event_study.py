"""
Script 08: Event Study — Extended CAR Analysis
Computes Cumulative Abnormal Returns over multiple windows:
  - Short:  [-3, +3]   (original)
  - Medium: [-3, +10]  (delayed institutional processing)
  - Long:   [-3, +30]  (slow-burn drift)
  - Pre:    [-30, -1]  (pre-filing information leakage)

Primary splits:
  1. High vs Low vagueness_ratio  (strongest signal from regressions)
  2. High vs Low risk_update_intensity

Key output: CAR drift plot from day -5 to day +30
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import wrds

warnings.filterwarnings("ignore")

RISK_FACTORS_DIR = Path("data/risk_factors")
DIFF_INDEX_CSV   = Path("outputs/all_diff_index.csv")
VARS_CSV         = Path("outputs/firm_year_variables.csv")
FIRMS_CSV        = Path("outputs/all_firms.csv")
OUTPUT_DIR       = Path("outputs/analysis/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EVENT_DATA_CSV   = Path("outputs/event_study_data.csv")

# Extended windows
DRIFT_WINDOW   = (-5, 30)     # full drift plot window
ESTIM_WINDOW   = (-120, -10)  # market model estimation

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})


# ── Filing date lookup ────────────────────────────────────────────────────────

def build_filing_date_lookup():
    lookup = {}
    for f in RISK_FACTORS_DIR.glob("*.txt"):
        parts = f.name.split("_")
        if len(parts) < 6:
            continue
        date_str  = parts[0]
        accession = parts[5].replace("__risk", "").replace(".txt", "")
        try:
            lookup[accession] = pd.to_datetime(date_str, format="%Y%m%d")
        except Exception:
            pass
    return lookup


def get_filing_dates(diff_index):
    import json
    lookup = build_filing_date_lookup()
    dates = []
    for _, row in diff_index.iterrows():
        try:
            with open(row["filepath"]) as f:
                d = json.load(f)
            acc = d.get("accession_new", "")
            dates.append(lookup.get(acc, lookup.get(acc.replace("-", ""))))
        except Exception:
            dates.append(None)
    diff_index = diff_index.copy()
    diff_index["filing_date"] = dates
    return diff_index


# ── WRDS daily returns ────────────────────────────────────────────────────────

def fetch_daily_returns(db, tickers, date_min, date_max):
    ticker_sql = ", ".join(f"'{t}'" for t in tickers)
    stocks = db.raw_sql(f"""
        SELECT a.permno, b.ticker, a.date, a.ret
        FROM crsp.dsf a
        JOIN crsp.dsenames b
          ON a.permno = b.permno
         AND a.date BETWEEN b.namedt AND COALESCE(b.nameendt, CURRENT_DATE)
        WHERE b.ticker IN ({ticker_sql})
          AND a.date BETWEEN '{date_min}' AND '{date_max}'
          AND a.ret IS NOT NULL
        ORDER BY b.ticker, a.date
    """, date_cols=["date"])

    market = db.raw_sql(f"""
        SELECT date, vwretd AS mkt_ret
        FROM crsp.dsi
        WHERE date BETWEEN '{date_min}' AND '{date_max}'
        ORDER BY date
    """, date_cols=["date"])

    return stocks.merge(market, on="date", how="left")


# ── CAR computation ───────────────────────────────────────────────────────────

def compute_car(daily, ticker, filing_date, drift_window, estim_window):
    """Compute AR for every day in drift_window. Returns dict of td_offset → CAR."""
    sub = daily[daily["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    if sub.empty:
        return None

    sub["td_diff"] = (sub["date"] - filing_date).dt.days
    event_candidates = sub[sub["td_diff"] >= 0]
    if event_candidates.empty:
        return None
    event_idx = event_candidates.index[0]
    sub["td_offset"] = range(-event_idx, len(sub) - event_idx)

    # Estimation window
    estim = sub[(sub["td_offset"] >= estim_window[0]) &
                (sub["td_offset"] <= estim_window[1])].dropna(subset=["ret", "mkt_ret"])
    if len(estim) < 30:
        return None

    from scipy.stats import linregress
    slope, intercept, *_ = linregress(estim["mkt_ret"], estim["ret"])

    # Full drift window
    event = sub[(sub["td_offset"] >= drift_window[0]) &
                (sub["td_offset"] <= drift_window[1])].dropna(subset=["ret", "mkt_ret"])
    if len(event) < 5:
        return None

    event = event.copy()
    event["ar"] = event["ret"] - (intercept + slope * event["mkt_ret"])

    car_series = {}
    cumulative = 0.0
    for _, erow in event.sort_values("td_offset").iterrows():
        cumulative += erow["ar"]
        car_series[int(erow["td_offset"])] = cumulative

    return {"ticker": ticker, "filing_date": filing_date,
            "car_series": car_series, "beta": slope}


# ── Plotting helpers ──────────────────────────────────────────────────────────

def plot_drift(df, split_col, high_label, low_label,
               color_high, color_low, title, filename, days):
    """Plot CAR drift for high vs low group with confidence bands."""
    car_cols = [f"car_d{d:+d}" for d in days if f"car_d{d:+d}" in df.columns]
    actual_days = [int(c.replace("car_d", "").replace("+", "")) for c in car_cols]

    high = df[df[split_col]][car_cols].mean()
    low  = df[~df[split_col]][car_cols].mean()
    high_se = df[df[split_col]][car_cols].sem()
    low_se  = df[~df[split_col]][car_cols].sem()

    n_high = df[split_col].sum()
    n_low  = (~df[split_col]).sum()

    # T-tests at key horizons
    results = {}
    for horizon, col in [("d0", "car_d+0"), ("d+5", "car_d+5"),
                          ("d+10", "car_d+10"), ("d+20", "car_d+20"), ("d+30", "car_d+30")]:
        if col in df.columns:
            t, p = stats.ttest_ind(
                df[df[split_col]][col].dropna(),
                df[~df[split_col]][col].dropna()
            )
            results[horizon] = (t, p)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Left: drift plot
    ax = axes[0]
    ax.plot(actual_days, high.values * 100, "-", color=color_high, linewidth=2.5,
            label=f"{high_label} (n={n_high})")
    ax.fill_between(actual_days,
                    (high - 1.96*high_se).values * 100,
                    (high + 1.96*high_se).values * 100,
                    alpha=0.15, color=color_high)
    ax.plot(actual_days, low.values * 100, "--", color=color_low, linewidth=2.5,
            label=f"{low_label} (n={n_low})")
    ax.fill_between(actual_days,
                    (low - 1.96*low_se).values * 100,
                    (low + 1.96*low_se).values * 100,
                    alpha=0.15, color=color_low)
    ax.axvline(0,  color="gray",  linestyle="--", linewidth=1.2, label="Filing date (day 0)")
    ax.axvline(-1, color="gray",  linestyle=":",  linewidth=0.8)
    ax.axhline(0,  color="black", linewidth=0.8)
    ax.set_xlabel("Trading Days Relative to 10-K Filing", fontsize=11)
    ax.set_ylabel("Cumulative Abnormal Return (%)", fontsize=11)
    ax.set_title("CAR Drift: Day -5 to Day +30", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xticks([-5, -3, 0, 3, 5, 10, 15, 20, 25, 30])

    # Right: significance table
    ax2 = axes[1]
    ax2.axis("off")
    rows = []
    for horizon, (t, p) in results.items():
        h_mean = df[df[split_col]][f"car_d{horizon.replace('d', '') if horizon != 'd0' else '+0'}"].mean() * 100 if f"car_d{horizon.replace('d', '') if horizon != 'd0' else '+0'}" in df.columns else np.nan
        l_mean = df[~df[split_col]][f"car_d{horizon.replace('d', '') if horizon != 'd0' else '+0'}"].mean() * 100 if f"car_d{horizon.replace('d', '') if horizon != 'd0' else '+0'}" in df.columns else np.nan
        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        rows.append([horizon, f"{h_mean:.3f}%", f"{l_mean:.3f}%", f"{t:.2f}", f"{p:.3f} {sig}"])

    table = ax2.table(
        cellText=rows,
        colLabels=["Horizon", f"{high_label} CAR", f"{low_label} CAR", "t-stat", "p-value"],
        loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    ax2.set_title("Statistical Tests at Key Horizons", fontsize=12, pad=20)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filename}")
    return results


def plot_pre_filing_drift(df, days):
    """Plot pre-filing drift to test information leakage."""
    pre_days = [d for d in days if d < 0]
    car_cols = [f"car_d{d:+d}" for d in pre_days if f"car_d{d:+d}" in df.columns]
    actual_days = [int(c.replace("car_d", "").replace("+", "")) for c in car_cols]

    if not car_cols:
        return

    med_vague = df["vagueness_ratio"].median()
    high_vague = df[df["vagueness_ratio"] > med_vague][car_cols].mean()
    low_vague  = df[df["vagueness_ratio"] <= med_vague][car_cols].mean()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(actual_days, high_vague.values * 100, "o-", color="#8e44ad",
            linewidth=2.2, label="High vagueness")
    ax.plot(actual_days, low_vague.values * 100, "s-", color="#27ae60",
            linewidth=2.2, label="Low vagueness")
    ax.axvline(0, color="gray", linestyle="--", linewidth=1.2, label="Filing date")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Trading Days Before 10-K Filing", fontsize=11)
    ax.set_ylabel("Cumulative Abnormal Return (%)", fontsize=11)
    ax.set_title("Pre-Filing CAR Drift: Information Leakage Test\n"
                 "(Negative pre-drift = market anticipates disclosure)", fontsize=12)
    ax.legend(fontsize=10)
    ax.invert_xaxis()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "14_pre_filing_drift.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved 14_pre_filing_drift.png")


def plot_yearly_cars(df, days):
    """Average CAR at +10 days by year — shows COVID effect."""
    col = "car_d+10" if "car_d+10" in df.columns else "car_d+3"
    yearly = df.groupby("year_new")[col].mean() * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in yearly.values]
    ax.bar(yearly.index, yearly.values, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Filing Year", fontsize=11)
    ax.set_ylabel(f"Average CAR at Day +10 (%)", fontsize=11)
    ax.set_title("Average Post-Filing CAR by Year\n(Red = negative drift, COVID years visible)",
                 fontsize=12)
    ax.set_xticks(yearly.index)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "15_car_by_year.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved 15_car_by_year.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    diff_index = pd.read_csv(DIFF_INDEX_CSV, dtype={"cik": str})
    vars_df    = pd.read_csv(VARS_CSV, dtype={"cik": str})
    firms      = pd.read_csv(FIRMS_CSV, dtype={"cik": str})

    print("Building filing date lookup from risk factor filenames...")
    diff_index = get_filing_dates(diff_index)
    print(f"  Filing dates found: {diff_index['filing_date'].notna().sum()} / {len(diff_index)}")

    diff_index = diff_index.merge(
        vars_df[["cik", "year_new", "risk_update_intensity", "vagueness_ratio",
                 "risk_financing", "risk_regulatory"]],
        on=["cik", "year_new"], how="left"
    )
    if "ticker" not in diff_index.columns:
        diff_index = diff_index.merge(firms[["cik", "ticker"]], on="cik", how="left")

    events = diff_index.dropna(subset=["filing_date", "risk_update_intensity", "ticker"])
    print(f"  Events with all data: {len(events)}")

    # Wider date range for extended windows
    min_date = (pd.to_datetime(events["filing_date"].min()) +
                pd.DateOffset(days=ESTIM_WINDOW[0] - 10)).strftime("%Y-%m-%d")
    max_date = (pd.to_datetime(events["filing_date"].max()) +
                pd.DateOffset(days=DRIFT_WINDOW[1] + 10)).strftime("%Y-%m-%d")

    tickers = events["ticker"].dropna().unique().tolist()
    print(f"Connecting to WRDS for daily returns ({min_date} → {max_date})...")
    db = wrds.Connection()
    daily = fetch_daily_returns(db, tickers, min_date, max_date)
    db.close()
    print(f"  Daily return rows fetched: {len(daily):,}")

    # Compute CARs
    print("Computing CARs (extended windows)...")
    results = []
    for _, row in events.iterrows():
        res = compute_car(daily, row["ticker"],
                          pd.to_datetime(row["filing_date"]),
                          DRIFT_WINDOW, ESTIM_WINDOW)
        if res is None:
            continue
        flat = {f"car_d{k:+d}": v for k, v in res["car_series"].items()}
        results.append({
            "ticker":               row["ticker"],
            "cik":                  row["cik"],
            "year_new":             row["year_new"],
            "filing_date":          row["filing_date"],
            "risk_update_intensity":row["risk_update_intensity"],
            "vagueness_ratio":      row["vagueness_ratio"],
            "risk_financing":       row["risk_financing"],
            "risk_regulatory":      row["risk_regulatory"],
            "beta":                 res["beta"],
            **flat,
        })

    df = pd.DataFrame(results)
    df.to_csv(EVENT_DATA_CSV, index=False)
    print(f"Event study dataset: {len(df)} events → {EVENT_DATA_CSV}")

    if df.empty:
        print("No events — check data.")
        return

    # Build split variables
    df["high_intensity"] = df["risk_update_intensity"] > df["risk_update_intensity"].median()
    df["high_vague"]     = df["vagueness_ratio"] > df["vagueness_ratio"].median()

    all_days = sorted(set(range(DRIFT_WINDOW[0], DRIFT_WINDOW[1] + 1)))

    # ── Plot 1: Vagueness drift (primary) ─────────────────────────────────────
    print("\n--- Vagueness Split ---")
    vague_results = plot_drift(
        df, "high_vague",
        "High Vagueness", "Low Vagueness",
        "#8e44ad", "#27ae60",
        "CAR Drift Around 10-K Filing: High vs Low Vagueness in Risk Disclosures",
        "12_event_study_car_vagueness.png", all_days
    )

    # ── Plot 2: Intensity drift ───────────────────────────────────────────────
    print("\n--- Intensity Split ---")
    intensity_results = plot_drift(
        df, "high_intensity",
        "High Intensity", "Low Intensity",
        "#e74c3c", "#2980b9",
        "CAR Drift Around 10-K Filing: High vs Low Risk Update Intensity",
        "13_event_study_car_intensity.png", all_days
    )

    # ── Plot 3: Pre-filing drift ──────────────────────────────────────────────
    plot_pre_filing_drift(df, all_days)

    # ── Plot 4: By year ───────────────────────────────────────────────────────
    plot_yearly_cars(df, all_days)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("EXTENDED EVENT STUDY SUMMARY")
    print(f"{'='*60}")
    print(f"Total events: {len(df)} | Years: {sorted(df['year_new'].unique())}")
    print(f"\nVAGUENESS SPLIT — CAR at key horizons:")
    for horizon, (t, p) in vague_results.items():
        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "(ns)"
        print(f"  {horizon}: t={t:.3f}, p={p:.3f} {sig}")
    print(f"\nINTENSITY SPLIT — CAR at key horizons:")
    for horizon, (t, p) in intensity_results.items():
        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "(ns)"
        print(f"  {horizon}: t={t:.3f}, p={p:.3f} {sig}")


if __name__ == "__main__":
    main()
