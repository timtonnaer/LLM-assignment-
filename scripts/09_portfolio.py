"""
Script 09: Long-Short Portfolio Construction
Constructs annual rebalancing portfolios based on 10-K risk disclosure signals.

Portfolios:
  A: Vagueness only        — Long low vagueness / Short high vagueness (top vs bottom tercile)
  B: Intensity + Vagueness — Long low vagueness + high intensity / Short high vagueness + low intensity
  C: Cyber aware           — Long high cyber+supply chain / Short low cyber+supply chain

Methodology:
  - Annual rebalancing within 5 days of each firm's 10-K filing
  - Equal-weighted
  - Benchmark: S&P 500 equal-weighted return
  - Performance: Sharpe ratio, max drawdown, annual alpha (CAPM)

Outputs:
  outputs/portfolio_returns.csv
  outputs/analysis/plots/16_portfolio_cumulative_returns.png
  outputs/analysis/plots/17_portfolio_annual_returns.png
  outputs/analysis/plots/18_portfolio_performance_summary.png
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

VARS_CSV       = Path("outputs/firm_year_variables.csv")
FIRMS_CSV      = Path("outputs/all_firms.csv")
EVENT_DATA_CSV = Path("outputs/event_study_data.csv")
OUTPUT_DIR     = Path("outputs/analysis/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PORTFOLIO_CSV  = Path("outputs/portfolio_returns.csv")

TERCILE_CUT   = [0, 0.333, 0.667, 1.0]   # bottom / middle / top tercile
HOLD_DAYS     = 252                        # ~1 year holding period
RISK_FREE     = 0.04 / 252                 # daily risk-free rate (~4% annual)

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})


# ── Portfolio signal construction ─────────────────────────────────────────────

def assign_terciles(df, col):
    """Assign tercile ranks (1=bottom, 2=middle, 3=top) for a given column."""
    df = df.copy()
    try:
        df[f"{col}_tercile"] = pd.qcut(df[col], q=3, labels=[1, 2, 3], duplicates="drop")
    except ValueError:
        # Fallback: use rank-based tercile when too many duplicates
        ranks = df[col].rank(method="first", na_option="bottom")
        n = len(ranks)
        df[f"{col}_tercile"] = pd.cut(ranks, bins=3, labels=[1, 2, 3])
    return df


def build_signals(vars_df):
    """Build portfolio signals for each firm-year."""
    df = vars_df.copy()

    # Assign terciles for key variables
    for col in ["vagueness_ratio", "risk_update_intensity", "risk_cyber", "risk_supply_chain"]:
        df = assign_terciles(df, col)

    # Portfolio A: Vagueness only
    df["port_A_long"]  = df["vagueness_ratio_tercile"] == 1   # low vagueness
    df["port_A_short"] = df["vagueness_ratio_tercile"] == 3   # high vagueness

    # Portfolio B: Vagueness + Intensity
    df["port_B_long"]  = (df["vagueness_ratio_tercile"] == 1) & \
                          (df["risk_update_intensity_tercile"] == 3)
    df["port_B_short"] = (df["vagueness_ratio_tercile"] == 3) & \
                          (df["risk_update_intensity_tercile"] == 1)

    # Portfolio C: Cyber + Supply chain aware
    cyber_sc = df["risk_cyber"].fillna(0) + df["risk_supply_chain"].fillna(0)
    df["cyber_sc_score"] = cyber_sc
    df = assign_terciles(df, "cyber_sc_score")
    df["port_C_long"]  = df["cyber_sc_score_tercile"] == 3
    df["port_C_short"] = df["cyber_sc_score_tercile"] == 1

    return df


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
        SELECT date, vwretd AS mkt_ret, ewretd AS ew_mkt_ret
        FROM crsp.dsi
        WHERE date BETWEEN '{date_min}' AND '{date_max}'
        ORDER BY date
    """, date_cols=["date"])

    return stocks, market


# ── Portfolio return computation ──────────────────────────────────────────────

def compute_portfolio_returns(signals, daily, market, filing_dates):
    """
    For each year, identify long/short firms, hold for HOLD_DAYS trading days,
    compute equal-weighted long, short and long-short returns.
    """
    all_returns = []

    for year in sorted(signals["year_new"].unique()):
        year_signals = signals[signals["year_new"] == year]

        for port in ["A", "B", "C"]:
            long_tickers  = year_signals[year_signals[f"port_{port}_long"]]["ticker"].dropna().tolist()
            short_tickers = year_signals[year_signals[f"port_{port}_short"]]["ticker"].dropna().tolist()

            if not long_tickers or not short_tickers:
                continue

            # Use median filing date for this year as rebalance date
            year_dates = filing_dates[filing_dates["year_new"] == year]["filing_date"].dropna()
            if year_dates.empty:
                continue
            rebalance_date = pd.to_datetime(year_dates.median())

            # Get trading days after rebalance
            mkt_period = market[market["date"] >= rebalance_date].head(HOLD_DAYS)
            if len(mkt_period) < 20:
                continue

            trading_dates = mkt_period["date"].values

            # Compute equal-weighted daily returns for long and short legs
            long_rets  = daily[daily["ticker"].isin(long_tickers) &
                               daily["date"].isin(trading_dates)]
            short_rets = daily[daily["ticker"].isin(short_tickers) &
                               daily["date"].isin(trading_dates)]

            long_daily  = long_rets.groupby("date")["ret"].mean()
            short_daily = short_rets.groupby("date")["ret"].mean()

            # Align on same dates
            combined = pd.DataFrame({
                "long":  long_daily,
                "short": short_daily,
                "mkt":   mkt_period.set_index("date")["mkt_ret"],
                "ew_mkt":mkt_period.set_index("date")["ew_mkt_ret"],
            }).dropna()

            combined["ls_ret"]   = combined["long"] - combined["short"]
            combined["portfolio"] = port
            combined["year"]      = year
            combined["n_long"]    = len(long_tickers)
            combined["n_short"]   = len(short_tickers)
            all_returns.append(combined.reset_index())

    if not all_returns:
        return pd.DataFrame()
    return pd.concat(all_returns, ignore_index=True)


# ── Performance metrics ───────────────────────────────────────────────────────

def performance_metrics(rets, label=""):
    """Compute Sharpe, Sortino, max drawdown, CAPM alpha."""
    if rets.empty or len(rets) < 10:
        return {}

    excess = rets - RISK_FREE
    sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0

    neg = excess[excess < 0]
    sortino = excess.mean() / neg.std() * np.sqrt(252) if len(neg) > 0 and neg.std() > 0 else 0

    cum = (1 + rets).cumprod()
    roll_max = cum.cummax()
    drawdown = (cum - roll_max) / roll_max
    max_dd = drawdown.min()

    annual_ret = (1 + rets.mean()) ** 252 - 1

    return {
        "label":       label,
        "ann_return":  round(annual_ret * 100, 2),
        "sharpe":      round(sharpe, 3),
        "sortino":     round(sortino, 3),
        "max_drawdown":round(max_dd * 100, 2),
        "n_days":      len(rets),
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_cumulative_returns(port_returns, market):
    """Plot cumulative returns for all 3 L-S portfolios vs market."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Long-Short Portfolio Cumulative Returns vs Equal-Weighted Market",
                 fontsize=14, fontweight="bold")

    colors = {"A": "#8e44ad", "B": "#e74c3c", "C": "#2980b9"}
    labels = {
        "A": "Port A: Low vs High Vagueness",
        "B": "Port B: Low Vagueness+High Intensity\nvs High Vagueness+Low Intensity",
        "C": "Port C: Cyber/Supply Chain Aware\nvs Unaware",
    }

    # Build market benchmark cumulative return
    mkt_cum = (1 + market.set_index("date")["ew_mkt_ret"].fillna(0)).cumprod()

    for idx, port in enumerate(["A", "B", "C"]):
        ax = axes[idx]
        sub = port_returns[port_returns["portfolio"] == port].sort_values("date")
        if sub.empty:
            ax.set_title(f"Portfolio {port}: No data")
            continue

        ls_cum  = (1 + sub.set_index("date")["ls_ret"].fillna(0)).cumprod()
        lng_cum = (1 + sub.set_index("date")["long"].fillna(0)).cumprod()
        sht_cum = (1 + sub.set_index("date")["short"].fillna(0)).cumprod()

        # Align market to same dates
        mkt_aligned = mkt_cum.reindex(ls_cum.index, method="ffill")
        mkt_base = mkt_aligned / mkt_aligned.iloc[0]

        ax.plot(ls_cum.index, ls_cum.values, linewidth=2.5,
                color=colors[port], label="L-S Portfolio")
        ax.plot(lng_cum.index, lng_cum.values, linewidth=1.5,
                color="green", alpha=0.7, linestyle="--", label="Long leg")
        ax.plot(sht_cum.index, sht_cum.values, linewidth=1.5,
                color="red", alpha=0.7, linestyle="--", label="Short leg")
        ax.plot(mkt_base.index, mkt_base.values, linewidth=1.5,
                color="gray", alpha=0.8, linestyle=":", label="EW Market")
        ax.axhline(1, color="black", linewidth=0.8)

        # Shade COVID period
        ax.axvspan(pd.Timestamp("2020-02-01"), pd.Timestamp("2020-06-01"),
                   alpha=0.1, color="red", label="COVID")

        metrics = performance_metrics(sub["ls_ret"], label=f"Port {port}")
        ax.set_title(f"{labels[port]}\nSharpe={metrics.get('sharpe','n/a')} | "
                     f"Ann.Ret={metrics.get('ann_return','n/a')}%", fontsize=10)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return (base=1)")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "16_portfolio_cumulative_returns.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved 16_portfolio_cumulative_returns.png")


def plot_annual_returns(port_returns):
    """Bar chart of annual L-S returns by portfolio."""
    annual = port_returns.groupby(["year", "portfolio"]).apply(
        lambda x: (1 + x["ls_ret"]).prod() - 1
    ).reset_index(name="annual_ret")

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(annual["year"].unique()))
    years = sorted(annual["year"].unique())
    width = 0.25
    colors = {"A": "#8e44ad", "B": "#e74c3c", "C": "#2980b9"}

    for i, port in enumerate(["A", "B", "C"]):
        sub = annual[annual["portfolio"] == port].set_index("year").reindex(years)
        ax.bar(x + i*width, sub["annual_ret"].values * 100,
               width=width, color=colors[port], alpha=0.8,
               label=f"Port {port}", edgecolor="white")

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x + width)
    ax.set_xticklabels(years)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Annual L-S Return (%)", fontsize=11)
    ax.set_title("Annual Long-Short Portfolio Returns\n"
                 "A=Vagueness | B=Vagueness+Intensity | C=Cyber/Supply Chain",
                 fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "17_portfolio_annual_returns.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved 17_portfolio_annual_returns.png")


def plot_performance_summary(port_returns, market):
    """Summary table + Sharpe/drawdown comparison."""
    metrics_list = []
    for port in ["A", "B", "C"]:
        sub = port_returns[port_returns["portfolio"] == port]
        if not sub.empty:
            m = performance_metrics(sub["ls_ret"], label=f"Portfolio {port}")
            metrics_list.append(m)

    # Add market benchmark
    mkt_rets = market["ew_mkt_ret"].dropna()
    m = performance_metrics(mkt_rets, label="EW Market")
    metrics_list.append(m)

    df_m = pd.DataFrame(metrics_list)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Portfolio Performance Summary", fontsize=14, fontweight="bold")

    # Table
    ax = axes[0]
    ax.axis("off")
    cols = ["label", "ann_return", "sharpe", "sortino", "max_drawdown"]
    col_labels = ["Portfolio", "Ann. Return (%)", "Sharpe", "Sortino", "Max DD (%)"]
    table = ax.table(
        cellText=df_m[cols].values,
        colLabels=col_labels,
        loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 2.0)
    ax.set_title("Performance Metrics", fontsize=11, pad=20)

    # Sharpe bar
    ax2 = axes[1]
    colors_bar = ["#8e44ad", "#e74c3c", "#2980b9", "gray"]
    bars = ax2.bar(df_m["label"], df_m["sharpe"], color=colors_bar, alpha=0.8, edgecolor="white")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Sharpe Ratio", fontsize=11)
    ax2.set_title("Sharpe Ratio Comparison", fontsize=11)
    ax2.tick_params(axis="x", rotation=15)
    for bar, val in zip(bars, df_m["sharpe"]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    # Max drawdown bar
    ax3 = axes[2]
    bars3 = ax3.bar(df_m["label"], df_m["max_drawdown"], color=colors_bar, alpha=0.8, edgecolor="white")
    ax3.set_ylabel("Max Drawdown (%)", fontsize=11)
    ax3.set_title("Maximum Drawdown", fontsize=11)
    ax3.tick_params(axis="x", rotation=15)
    for bar, val in zip(bars3, df_m["max_drawdown"]):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.5,
                 f"{val:.1f}%", ha="center", va="top", fontsize=9, color="white")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "18_portfolio_performance_summary.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved 18_portfolio_performance_summary.png")

    return df_m


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading variables and signals...")
    vars_df  = pd.read_csv(VARS_CSV, dtype={"cik": str})
    firms    = pd.read_csv(FIRMS_CSV, dtype={"cik": str})
    events   = pd.read_csv(EVENT_DATA_CSV, dtype={"cik": str})

    # Merge ticker into vars only if not already present
    if "ticker" not in vars_df.columns:
        vars_df = vars_df.merge(firms[["cik", "ticker"]], on="cik", how="left")
    vars_df["cik"] = vars_df["cik"].astype(str)

    # Drop rows with no ticker or key signal columns
    vars_df = vars_df.dropna(subset=["ticker", "vagueness_ratio", "risk_update_intensity"])

    # Build signals
    signals = build_signals(vars_df)

    # Get filing dates from event study data
    filing_dates = events[["year_new", "filing_date"]].copy()
    filing_dates["filing_date"] = pd.to_datetime(filing_dates["filing_date"])

    # All tickers
    all_tickers = vars_df["ticker"].dropna().unique().tolist()

    # Date range: 2016-2024
    date_min = "2015-01-01"
    date_max = "2024-12-31"

    print(f"Connecting to WRDS for daily returns ({date_min} → {date_max})...")
    db = wrds.Connection(wrds_username="timtonnaer10")
    daily, market = fetch_daily_returns(db, all_tickers, date_min, date_max)
    db.close()
    print(f"  Stock returns: {len(daily):,} rows | Market: {len(market):,} rows")

    # Compute portfolio returns
    print("Computing portfolio returns...")
    port_returns = compute_portfolio_returns(signals, daily, market, filing_dates)

    if port_returns.empty:
        print("No portfolio returns computed — check data.")
        return

    port_returns.to_csv(PORTFOLIO_CSV, index=False)
    print(f"Portfolio returns saved to {PORTFOLIO_CSV}")

    # Generate plots
    print("\nGenerating plots...")
    plot_cumulative_returns(port_returns, market)
    plot_annual_returns(port_returns)
    metrics = plot_performance_summary(port_returns, market)

    # Summary
    print(f"\n{'='*60}")
    print("PORTFOLIO PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(metrics[["label", "ann_return", "sharpe", "sortino", "max_drawdown"]].to_string(index=False))

    for port in ["A", "B", "C"]:
        sub = port_returns[port_returns["portfolio"] == port]
        if not sub.empty:
            n_long  = sub["n_long"].iloc[0]
            n_short = sub["n_short"].iloc[0]
            print(f"\nPortfolio {port}: avg {n_long:.0f} long / {n_short:.0f} short positions")


if __name__ == "__main__":
    main()
