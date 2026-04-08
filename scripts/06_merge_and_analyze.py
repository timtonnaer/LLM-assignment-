"""
Script 06: Merge and Analyze
Builds the final panel dataset, runs predictive regressions, and generates visualizations.

Predictive design: risk disclosure variables at year t predict outcomes at year t+1.

Regressions:
  1. Future ROA         ~ risk_update_intensity + vagueness_ratio + controls + sector_FE
  2. Future rev_growth  ~ risk_update_intensity + vagueness_ratio + controls + sector_FE
  3. Future annual_ret  ~ risk_update_intensity + vagueness_ratio + controls + sector_FE
  4. Future ROA         ~ risk type counts + controls + sector_FE

Visualizations:
  1. Time series: avg risk_update_intensity per year
  2. Bar chart: avg risk type counts by sector
  3. Scatter: vagueness_ratio vs future ROA (colored by sector)
  4. Correlation heatmap of key variables
"""

import math
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from pathlib import Path

warnings.filterwarnings("ignore")

VARS_CSV = Path("/Users/timtonnaer/risk_project/outputs/firm_year_variables.csv")
COMPUSTAT_CSV = Path("/Users/timtonnaer/risk_project/outputs/compustat_panel.csv")
CRSP_CSV = Path("/Users/timtonnaer/risk_project/outputs/crsp_returns.csv")
FINAL_PANEL_CSV = Path("/Users/timtonnaer/risk_project/outputs/final_panel.csv")
RESULTS_TXT = Path("/Users/timtonnaer/risk_project/outputs/analysis/regression_results.txt")
PLOTS_DIR = Path("/Users/timtonnaer/risk_project/outputs/analysis/plots")


# ── Helpers ──────────────────────────────────────────────────────────────────

def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    lo, hi = series.quantile([lower, upper])
    return series.clip(lo, hi)


def sector_dummies(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df["sector"], prefix="sec", drop_first=True)


# ── Build Panel ───────────────────────────────────────────────────────────────

def build_panel() -> pd.DataFrame:
    text = pd.read_csv(VARS_CSV, dtype={"cik": str})
    compustat = pd.read_csv(COMPUSTAT_CSV, dtype={"cik": str})
    crsp = pd.read_csv(CRSP_CSV)

    # Normalize CIK formats: Compustat uses zero-padded 10-digit strings, strip to match text panel
    compustat["cik"] = compustat["cik"].str.lstrip("0")

    # Merge Compustat into text panel on (cik, year_new == fyear) for current controls
    panel = text.merge(
        compustat[["cik", "fyear", "roa", "rev_growth", "leverage", "current_ratio", "log_assets"]],
        left_on=["cik", "year_new"],
        right_on=["cik", "fyear"],
        how="left",
    ).drop(columns=["fyear"])

    # Merge t+1 outcomes from Compustat
    panel = panel.merge(
        compustat[["cik", "fyear", "roa", "rev_growth", "leverage", "current_ratio"]].rename(columns={
            "fyear": "fyear_t1",
            "roa": "roa_t1",
            "rev_growth": "rev_growth_t1",
            "leverage": "leverage_t1",
            "current_ratio": "current_ratio_t1",
        }),
        left_on=["cik", "year_new"],
        right_on=["cik", "fyear_t1"],
        suffixes=("", "_drop"),
        how="left",
    )
    # fyear_t1 should equal year_new + 1 by construction after shift
    # Re-do correctly: join on year_new+1
    panel = panel.drop(columns=[c for c in panel.columns if c.endswith("_drop") or c == "fyear_t1"])

    t1_outcomes = compustat[["cik", "fyear", "roa", "rev_growth", "leverage", "current_ratio"]].copy()
    t1_outcomes = t1_outcomes.rename(columns={
        "fyear": "year_t1",
        "roa": "roa_t1",
        "rev_growth": "rev_growth_t1",
        "leverage": "leverage_t1",
        "current_ratio": "current_ratio_t1",
    })
    t1_outcomes["year_new"] = t1_outcomes["year_t1"] - 1

    panel = panel.drop(columns=["roa_t1", "rev_growth_t1", "leverage_t1", "current_ratio_t1"], errors="ignore")
    panel = panel.merge(
        t1_outcomes[["cik", "year_new", "roa_t1", "rev_growth_t1", "leverage_t1", "current_ratio_t1"]],
        on=["cik", "year_new"],
        how="left",
    )

    # Merge CRSP returns (match on ticker + year)
    crsp_annual = crsp[["ticker", "year", "annual_ret"]].copy()
    crsp_annual["year"] = crsp_annual["year"].astype(int)

    # Current year return
    panel = panel.merge(
        crsp_annual.rename(columns={"year": "year_new", "annual_ret": "annual_ret"}),
        on=["ticker", "year_new"],
        how="left",
    )

    # t+1 return
    crsp_t1 = crsp_annual.copy()
    crsp_t1["year_new"] = crsp_t1["year"] - 1
    crsp_t1 = crsp_t1.rename(columns={"annual_ret": "annual_ret_t1"}).drop(columns=["year"])
    panel = panel.merge(crsp_t1, on=["ticker", "year_new"], how="left")

    # Winsorize continuous variables
    for col in ["roa_t1", "rev_growth_t1", "leverage", "risk_update_intensity",
                "vagueness_ratio", "boilerplate_ratio", "annual_ret_t1"]:
        if col in panel.columns:
            panel[col] = winsorize(panel[col].dropna().reindex(panel.index))

    return panel


# ── Regressions ───────────────────────────────────────────────────────────────

def run_regressions(panel: pd.DataFrame) -> str:
    lines = []

    # Clean sector names for use as column names (remove spaces/slashes)
    df = panel.copy()
    df["sector_clean"] = df["sector"].str.replace(r"[^a-zA-Z0-9]", "_", regex=True)

    # Create sector dummies (drop_first avoids multicollinearity)
    sector_dummies = pd.get_dummies(df["sector_clean"], prefix="sec", drop_first=True)
    df = pd.concat([df, sector_dummies], axis=1)
    sec_cols = " + ".join(sector_dummies.columns.tolist())
    controls = f"leverage + log_assets + {sec_cols}"

    specs = [
        ("roa_t1",        "risk_update_intensity + vagueness_ratio",
                          "Reg 1: Future ROA ~ Risk Update Intensity + Vagueness"),
        ("rev_growth_t1", "risk_update_intensity + vagueness_ratio",
                          "Reg 2: Future Revenue Growth ~ Risk Update Intensity + Vagueness"),
        ("annual_ret_t1", "risk_update_intensity + vagueness_ratio",
                          "Reg 3: Future Stock Return ~ Risk Update Intensity + Vagueness"),
        ("roa_t1",        "risk_financing + risk_operational + risk_cyber + risk_regulatory + risk_supply_chain",
                          "Reg 4: Future ROA ~ Risk Type Counts"),
    ]

    for dep, regressors, title in specs:
        reg_vars = [v.strip() for v in regressors.split("+")]
        ctrl_vars = ["leverage", "log_assets"] + sector_dummies.columns.tolist()
        all_vars = [dep] + reg_vars + ctrl_vars
        all_vars = [v for v in all_vars if v in df.columns]
        sub = df[all_vars].dropna()
        if len(sub) < 20:
            lines.append(f"\n{'='*60}\n{title}\nInsufficient data (n={len(sub)}), skipping.\n")
            continue
        try:
            formula = f"{dep} ~ {regressors} + {controls}"
            model = smf.ols(formula, data=sub).fit(cov_type="HC3")
            lines.append(f"\n{'='*60}\n{title}\n{model.summary().as_text()}\n")
            lines.append(f"N = {int(model.nobs)}, R² = {model.rsquared:.3f}, Adj-R² = {model.rsquared_adj:.3f}\n")
        except Exception as e:
            lines.append(f"\n{title}\nError: {e}\n")

    return "\n".join(lines)


# ── Visualizations ────────────────────────────────────────────────────────────

def make_plots(panel: pd.DataFrame):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Time series: avg risk_update_intensity per year
    fig, ax = plt.subplots(figsize=(10, 5))
    ts = panel.groupby("year_new")["risk_update_intensity"].mean().reset_index()
    ax.plot(ts["year_new"], ts["risk_update_intensity"], marker="o", linewidth=2)
    ax.set_title("Average Risk Update Intensity per Year (Pilot Firms)", fontsize=13)
    ax.set_xlabel("Year")
    ax.set_ylabel("Avg. Meaningful Risk Updates")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "01_risk_intensity_timeseries.png", dpi=150)
    plt.close(fig)

    # 2. Bar chart: avg risk type counts by sector
    risk_cols = [c for c in panel.columns if c.startswith("risk_") and c != "risk_update_intensity"]
    sector_risk = panel.groupby("sector")[risk_cols].mean()
    sector_risk.columns = [c.replace("risk_", "").replace("_", " ").title() for c in sector_risk.columns]
    fig, ax = plt.subplots(figsize=(12, 6))
    sector_risk.T.plot(kind="bar", ax=ax, width=0.7)
    ax.set_title("Average Risk Type Counts by Sector", fontsize=13)
    ax.set_xlabel("Risk Type")
    ax.set_ylabel("Avg. Count per Firm-Year")
    ax.legend(title="Sector", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "02_risk_types_by_sector.png", dpi=150)
    plt.close(fig)

    # 3. Scatter: vagueness_ratio vs future ROA
    sub = panel[["vagueness_ratio", "roa_t1", "sector", "ticker"]].dropna()
    if len(sub) > 5:
        fig, ax = plt.subplots(figsize=(9, 6))
        sectors = sub["sector"].unique()
        colors = sns.color_palette("tab10", len(sectors))
        for i, sec in enumerate(sectors):
            d = sub[sub["sector"] == sec]
            ax.scatter(d["vagueness_ratio"], d["roa_t1"], label=sec, alpha=0.7,
                       color=colors[i], s=50)
        # Trend line
        x, y = sub["vagueness_ratio"].values, sub["roa_t1"].values
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() > 2:
            z = np.polyfit(x[mask], y[mask], 1)
            p = np.poly1d(z)
            xs = np.linspace(x[mask].min(), x[mask].max(), 100)
            ax.plot(xs, p(xs), "k--", linewidth=1.5, label="Trend")
        ax.set_title("Vagueness of Risk Updates vs Future ROA", fontsize=13)
        ax.set_xlabel("Vagueness Ratio (year t)")
        ax.set_ylabel("ROA (year t+1)")
        ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
        plt.tight_layout()
        fig.savefig(PLOTS_DIR / "03_vagueness_vs_future_roa.png", dpi=150)
        plt.close(fig)

    # 4. Correlation heatmap
    corr_cols = ["risk_update_intensity", "vagueness_ratio", "boilerplate_ratio",
                 "similarity", "roa_t1", "rev_growth_t1", "leverage", "annual_ret_t1",
                 "risk_cyber", "risk_operational", "risk_financing", "risk_regulatory"]
    corr_cols = [c for c in corr_cols if c in panel.columns]
    corr_df = panel[corr_cols].dropna(how="all")
    if len(corr_df) > 5:
        corr_matrix = corr_df.corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            linewidths=0.5, ax=ax, annot_kws={"size": 8},
            vmin=-1, vmax=1,
        )
        labels = [c.replace("_", "\n") for c in corr_matrix.columns]
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, rotation=0, fontsize=8)
        ax.set_title("Correlation Matrix — Key Research Variables", fontsize=13)
        plt.tight_layout()
        fig.savefig(PLOTS_DIR / "04_correlation_heatmap.png", dpi=150)
        plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Building panel...")
    panel = build_panel()
    panel.to_csv(FINAL_PANEL_CSV, index=False)
    print(f"Final panel: {len(panel)} rows, {panel.columns.tolist()}")

    classified_rows = panel[panel["n_classified"] > 0]
    print(f"Rows with LLM classifications: {len(classified_rows)}")
    print(f"Rows with t+1 ROA outcome:     {panel['roa_t1'].notna().sum()}")
    print(f"Rows with t+1 return outcome:  {panel['annual_ret_t1'].notna().sum()}")

    RESULTS_TXT.parent.mkdir(parents=True, exist_ok=True)

    if len(classified_rows) < 20:
        msg = (
            "Insufficient classified rows for regression analysis.\n"
            "Run script 03 (LLM classification) first, then re-run this script.\n"
            f"Current classified rows: {len(classified_rows)}"
        )
        print(f"\n{msg}")
        RESULTS_TXT.write_text(msg)
    else:
        print("\nRunning regressions...")
        results = run_regressions(panel)
        RESULTS_TXT.write_text(results)
        print(f"Regression results saved to {RESULTS_TXT}")

    print("\nGenerating visualizations...")
    make_plots(panel)
    print(f"Plots saved to {PLOTS_DIR}/")

    print(f"\nFinal panel saved to {FINAL_PANEL_CSV}")


if __name__ == "__main__":
    main()
