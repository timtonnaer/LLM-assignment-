"""
Script 08: Event Study — Stock Price Reaction to 10-K Filings
Computes Cumulative Abnormal Returns (CARs) in a [-3, +3] trading-day window
around each 10-K filing date, comparing firms with high vs low risk disclosure changes.

Methodology:
  - Filing dates: extracted from risk factor filenames (YYYYMMDD_10-K_..._accession__risk.txt)
  - Normal returns: market model estimated over [-120, -10] trading days pre-filing
  - Abnormal return: AR_t = R_t - (alpha + beta * R_market_t)
  - CAR = cumulative sum of AR over event window

Comparisons:
  1. High vs Low risk_update_intensity
  2. High vs Low vagueness_ratio
  3. Financing risk disclosers vs non-disclosers
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
DIFF_INDEX_CSV   = Path("outputs/pilot_diff_index.csv")
VARS_CSV         = Path("outputs/firm_year_variables.csv")
PILOT_FIRMS_CSV  = Path("outputs/pilot_firms.csv")
OUTPUT_DIR       = Path("outputs/analysis/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EVENT_DATA_CSV   = Path("outputs/event_study_data.csv")

EVENT_WINDOW   = (-3, 3)      # trading days around filing
ESTIM_WINDOW   = (-120, -10)  # trading days for beta estimation

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})


# ── Step 1: Build filing date lookup from risk factor filenames ───────────────

def build_filing_date_lookup() -> dict:
    """Returns {accession_number: filing_date} from risk factor filenames."""
    lookup = {}
    for f in RISK_FACTORS_DIR.glob("*.txt"):
        parts = f.name.split("_")
        if len(parts) < 6:
            continue
        date_str = parts[0]          # YYYYMMDD
        accession = parts[5].replace("__risk", "").replace(".txt", "")
        try:
            filing_date = pd.to_datetime(date_str, format="%Y%m%d")
            lookup[accession] = filing_date
        except Exception:
            pass
    return lookup


def get_filing_dates(diff_index: pd.DataFrame) -> pd.DataFrame:
    """Add filing_date column to diff_index using risk factor filename lookup."""
    import json
    lookup = build_filing_date_lookup()

    dates = []
    for _, row in diff_index.iterrows():
        diff_path = row["filepath"]
        try:
            with open(diff_path) as f:
                d = json.load(f)
            accession = d.get("accession_new", "")
            # Normalize: remove dashes for lookup
            accession_clean = accession.replace("-", "")
            filing_date = lookup.get(accession, lookup.get(accession_clean))
            dates.append(filing_date)
        except Exception:
            dates.append(None)

    diff_index = diff_index.copy()
    diff_index["filing_date"] = dates
    return diff_index


# ── Step 2: Fetch daily returns from WRDS ────────────────────────────────────

def fetch_daily_returns(db, tickers: list[str],
                        date_min: str, date_max: str) -> pd.DataFrame:
    """Pull daily stock returns and market return from CRSP."""
    ticker_sql = ", ".join(f"'{t}'" for t in tickers)

    # Daily stock returns
    stock_q = f"""
        SELECT a.permno, b.ticker, a.date, a.ret
        FROM crsp.dsf a
        JOIN crsp.dsenames b
          ON a.permno = b.permno
         AND a.date BETWEEN b.namedt AND COALESCE(b.nameendt, CURRENT_DATE)
        WHERE b.ticker IN ({ticker_sql})
          AND a.date BETWEEN '{date_min}' AND '{date_max}'
          AND a.ret IS NOT NULL
        ORDER BY b.ticker, a.date
    """
    stocks = db.raw_sql(stock_q, date_cols=["date"])

    # Market return (CRSP value-weighted index)
    mkt_q = f"""
        SELECT date, vwretd AS mkt_ret
        FROM crsp.dsi
        WHERE date BETWEEN '{date_min}' AND '{date_max}'
        ORDER BY date
    """
    market = db.raw_sql(mkt_q, date_cols=["date"])

    stocks = stocks.merge(market, on="date", how="left")
    return stocks


# ── Step 3: Compute CARs for each event ──────────────────────────────────────

def compute_car(daily: pd.DataFrame, ticker: str, filing_date: pd.Timestamp,
                event_window: tuple, estim_window: tuple) -> dict | None:
    """
    Compute CAR for one filing event.
    Returns dict with CAR at each day in event window plus metadata.
    """
    sub = daily[daily["ticker"] == ticker].sort_values("date").copy()
    if sub.empty:
        return None

    # Build a trading-day calendar relative to filing date
    # Find the index of the nearest trading day to the filing date
    sub = sub.reset_index(drop=True)
    sub["td_diff"] = (sub["date"] - filing_date).dt.days

    # Find the event day (closest trading day on or after filing)
    event_candidates = sub[sub["td_diff"] >= 0]
    if event_candidates.empty:
        return None
    event_idx = event_candidates.index[0]

    # Map to trading-day offsets
    sub["td_offset"] = range(-event_idx, len(sub) - event_idx)

    # Estimation window
    estim = sub[(sub["td_offset"] >= estim_window[0]) &
                (sub["td_offset"] <= estim_window[1])].dropna(subset=["ret", "mkt_ret"])
    if len(estim) < 30:
        return None

    # OLS market model: R_i = alpha + beta * R_m
    from scipy.stats import linregress
    slope, intercept, *_ = linregress(estim["mkt_ret"], estim["ret"])

    # Event window
    event = sub[(sub["td_offset"] >= event_window[0]) &
                (sub["td_offset"] <= event_window[1])].dropna(subset=["ret", "mkt_ret"])

    if len(event) < (event_window[1] - event_window[0]):
        return None

    # Abnormal returns
    event = event.copy()
    event["expected_ret"] = intercept + slope * event["mkt_ret"]
    event["ar"] = event["ret"] - event["expected_ret"]

    # Build CAR series keyed by td_offset
    car_series = {}
    cumulative = 0.0
    for _, erow in event.sort_values("td_offset").iterrows():
        cumulative += erow["ar"]
        car_series[int(erow["td_offset"])] = cumulative

    return {"ticker": ticker, "filing_date": filing_date, "car_series": car_series,
            "alpha": intercept, "beta": slope}


# ── Step 4: Main ──────────────────────────────────────────────────────────────

def main():
    diff_index = pd.read_csv(DIFF_INDEX_CSV, dtype={"cik": str})
    vars_df    = pd.read_csv(VARS_CSV, dtype={"cik": str})
    firms      = pd.read_csv(PILOT_FIRMS_CSV, dtype={"cik": str})

    print("Building filing date lookup from risk factor filenames...")
    diff_index = get_filing_dates(diff_index)
    n_dates = diff_index["filing_date"].notna().sum()
    print(f"  Filing dates found: {n_dates} / {len(diff_index)}")

    # Merge in text variables
    diff_index = diff_index.merge(
        vars_df[["cik", "year_new", "risk_update_intensity", "vagueness_ratio",
                 "risk_financing", "risk_regulatory"]],
        on=["cik", "year_new"], how="left"
    )
    # ticker is already in diff_index; merge firms only for any missing values
    if "ticker" not in diff_index.columns:
        diff_index = diff_index.merge(firms[["cik", "ticker"]], on="cik", how="left")

    # Keep only events with filing dates and text variables
    events = diff_index.dropna(subset=["filing_date", "risk_update_intensity", "ticker"])
    print(f"  Events with all data: {len(events)}")

    # Date range for WRDS pull
    min_date = (pd.to_datetime(events["filing_date"].min()) +
                pd.DateOffset(days=ESTIM_WINDOW[0] - 10)).strftime("%Y-%m-%d")
    max_date = (pd.to_datetime(events["filing_date"].max()) +
                pd.DateOffset(days=EVENT_WINDOW[1] + 5)).strftime("%Y-%m-%d")

    tickers = events["ticker"].dropna().unique().tolist()
    print(f"Connecting to WRDS for daily returns ({min_date} → {max_date})...")
    db = wrds.Connection()
    daily = fetch_daily_returns(db, tickers, min_date, max_date)
    db.close()
    print(f"  Daily return rows fetched: {len(daily):,}")

    # Compute CAR for each event
    print("Computing CARs...")
    results = []
    for _, row in events.iterrows():
        res = compute_car(
            daily, row["ticker"],
            pd.to_datetime(row["filing_date"]),
            EVENT_WINDOW, ESTIM_WINDOW
        )
        if res is None:
            continue
        # Flatten car_series into columns
        flat = {f"car_d{k:+d}": v for k, v in res["car_series"].items()}
        results.append({
            "ticker": row["ticker"],
            "cik": row["cik"],
            "year_new": row["year_new"],
            "filing_date": row["filing_date"],
            "risk_update_intensity": row["risk_update_intensity"],
            "vagueness_ratio": row["vagueness_ratio"],
            "risk_financing": row["risk_financing"],
            "risk_regulatory": row["risk_regulatory"],
            "beta": res["beta"],
            **flat,
        })

    df = pd.DataFrame(results)
    df.to_csv(EVENT_DATA_CSV, index=False)
    print(f"Event study dataset: {len(df)} events → {EVENT_DATA_CSV}")

    if df.empty:
        print("No events computed — check data coverage.")
        return

    # ── Plot 1: Average CAR — High vs Low Risk Update Intensity ──────────────
    median_intensity = df["risk_update_intensity"].median()
    df["high_intensity"] = df["risk_update_intensity"] > median_intensity

    car_cols = sorted([c for c in df.columns if c.startswith("car_d")],
                      key=lambda x: int(x.replace("car_d", "").replace("+", "").replace("-", "-")))
    days = [int(c.replace("car_d", "").replace("+", "")) for c in car_cols]

    high = df[df["high_intensity"]][car_cols].mean()
    low  = df[~df["high_intensity"]][car_cols].mean()
    high_se = df[df["high_intensity"]][car_cols].sem()
    low_se  = df[~df["high_intensity"]][car_cols].sem()

    # T-test at filing day (day 0 CAR)
    car0_col = "car_d+0" if "car_d+0" in df.columns else car_cols[len(car_cols)//2]
    t_stat, p_val = stats.ttest_ind(
        df[df["high_intensity"]][car0_col].dropna(),
        df[~df["high_intensity"]][car0_col].dropna()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Event Study: Stock Price Reaction to 10-K Filings (±3 Trading Days)",
                 fontsize=14, fontweight="bold")

    # Left panel: High vs Low intensity
    ax = axes[0]
    ax.plot(days, high.values * 100, "o-", color="#e74c3c", linewidth=2.2,
            label=f"High risk updates (n={df['high_intensity'].sum()})")
    ax.fill_between(days,
                    (high - 1.96*high_se).values * 100,
                    (high + 1.96*high_se).values * 100,
                    alpha=0.15, color="#e74c3c")
    ax.plot(days, low.values * 100, "s-", color="#2980b9", linewidth=2.2,
            label=f"Low risk updates (n={(~df['high_intensity']).sum()})")
    ax.fill_between(days,
                    (low - 1.96*low_se).values * 100,
                    (low + 1.96*low_se).values * 100,
                    alpha=0.15, color="#2980b9")
    ax.axvline(0, color="gray", linestyle="--", linewidth=1.2, label="Filing date")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Trading Days Relative to 10-K Filing", fontsize=11)
    ax.set_ylabel("Cumulative Abnormal Return (%)", fontsize=11)
    ax.set_title(f"High vs Low Risk Update Intensity\n(t={t_stat:.2f}, p={p_val:.3f})", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xticks(days)

    # Right panel: High vs Low vagueness
    med_vague = df["vagueness_ratio"].median()
    df["high_vague"] = df["vagueness_ratio"] > med_vague
    vague_h = df[df["high_vague"]][car_cols].mean()
    vague_l = df[~df["high_vague"]][car_cols].mean()
    vague_h_se = df[df["high_vague"]][car_cols].sem()
    vague_l_se = df[~df["high_vague"]][car_cols].sem()

    t2, p2 = stats.ttest_ind(
        df[df["high_vague"]][car0_col].dropna(),
        df[~df["high_vague"]][car0_col].dropna()
    )

    ax2 = axes[1]
    ax2.plot(days, vague_h.values * 100, "o-", color="#8e44ad", linewidth=2.2,
             label=f"High vagueness (n={df['high_vague'].sum()})")
    ax2.fill_between(days,
                     (vague_h - 1.96*vague_h_se).values * 100,
                     (vague_h + 1.96*vague_h_se).values * 100,
                     alpha=0.15, color="#8e44ad")
    ax2.plot(days, vague_l.values * 100, "s-", color="#27ae60", linewidth=2.2,
             label=f"Low vagueness (n={(~df['high_vague']).sum()})")
    ax2.fill_between(days,
                     (vague_l - 1.96*vague_l_se).values * 100,
                     (vague_l + 1.96*vague_l_se).values * 100,
                     alpha=0.15, color="#27ae60")
    ax2.axvline(0, color="gray", linestyle="--", linewidth=1.2, label="Filing date")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Trading Days Relative to 10-K Filing", fontsize=11)
    ax2.set_ylabel("Cumulative Abnormal Return (%)", fontsize=11)
    ax2.set_title(f"High vs Low Vagueness Ratio\n(t={t2:.2f}, p={p2:.3f})", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.set_xticks(days)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "12_event_study_car.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved 12_event_study_car.png")

    # ── Plot 2: CAR distribution at day 0 ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Distribution of CARs at Filing Date (Day 0)", fontsize=13, fontweight="bold")

    for ax, col, label, colors in [
        (axes[0], "high_intensity", "Risk Update Intensity", ("#e74c3c", "#2980b9")),
        (axes[1], "high_vague",     "Vagueness Ratio",       ("#8e44ad", "#27ae60")),
    ]:
        high_vals = df[df[col]][car0_col].dropna() * 100
        low_vals  = df[~df[col]][car0_col].dropna() * 100
        ax.hist(high_vals, bins=20, alpha=0.6, color=colors[0], label="High", edgecolor="white")
        ax.hist(low_vals,  bins=20, alpha=0.6, color=colors[1], label="Low",  edgecolor="white")
        ax.axvline(high_vals.mean(), color=colors[0], linestyle="--", linewidth=1.8,
                   label=f"High mean: {high_vals.mean():.2f}%")
        ax.axvline(low_vals.mean(),  color=colors[1], linestyle="--", linewidth=1.8,
                   label=f"Low mean: {low_vals.mean():.2f}%")
        ax.set_xlabel("CAR at Filing Date (%)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"By {label}", fontsize=12)
        ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "13_event_study_car_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved 13_event_study_car_distribution.png")

    # ── Summary stats ─────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("EVENT STUDY SUMMARY")
    print(f"{'='*50}")
    print(f"Total events analyzed: {len(df)}")
    print(f"Median intensity split: {median_intensity:.1f} meaningful updates")
    print(f"\nCAR at filing date (day 0):")
    print(f"  High intensity: {df[df['high_intensity']][car0_col].mean()*100:.3f}% "
          f"(n={df['high_intensity'].sum()})")
    print(f"  Low intensity:  {df[~df['high_intensity']][car0_col].mean()*100:.3f}% "
          f"(n={(~df['high_intensity']).sum()})")
    print(f"  Difference t-test: t={t_stat:.3f}, p={p_val:.3f}")
    print(f"\n  High vagueness: {df[df['high_vague']][car0_col].mean()*100:.3f}%")
    print(f"  Low vagueness:  {df[~df['high_vague']][car0_col].mean()*100:.3f}%")
    print(f"  Difference t-test: t={t2:.3f}, p={p2:.3f}")


if __name__ == "__main__":
    main()
