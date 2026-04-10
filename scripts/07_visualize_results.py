"""
Script 07: Visualize Results
Comprehensive visual representations of the risk disclosure research findings.
Outputs 10 publication-quality plots to outputs/analysis/plots/.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

PANEL_CSV   = Path("outputs/final_panel.csv")
PLOTS_DIR   = Path("outputs/analysis/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE   = sns.color_palette("tab10")
SECTOR_COLORS = {
    "Consumer Discretionary": PALETTE[0],
    "Financials":              PALETTE[1],
    "Health Care":             PALETTE[2],
    "Industrials":             PALETTE[3],
    "Information Technology":  PALETTE[4],
    "Consumer Staples":        "#e377c2",
    "Energy":                  "#8c564b",
    "Materials":               "#bcbd22",
    "Real Estate":             "#17becf",
    "Utilities":               "#9467bd",
    "Communication Services":  "#7f7f7f",
}
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})


def save(fig, name):
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(PANEL_CSV, dtype={"cik": str})
classified = df[df["n_classified"] > 0].copy()


# ════════════════════════════════════════════════════════════════════
# 1. Regression Coefficient Chart — Reg 4 (Risk Type → Future ROA)
# ════════════════════════════════════════════════════════════════════
coefs = {
    "Financing Risk":    (-0.0032, 0.00157),
    "Regulatory Risk":   (-0.0019, 0.00086),
    "Cyber Risk":        (-0.0012, 0.00190),
    "Operational Risk":  ( 0.0020, 0.00078),
    "Supply Chain Risk": ( 0.0053, 0.00259),
}
labels = list(coefs.keys())
vals   = [v[0] for v in coefs.values()]
errs   = [v[1]*1.96 for v in coefs.values()]   # 95% CI
colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in vals]

fig, ax = plt.subplots(figsize=(9, 5))
y_pos = range(len(labels))
ax.barh(y_pos, vals, xerr=errs, color=colors, alpha=0.85,
        error_kw={"ecolor": "black", "capsize": 5, "linewidth": 1.5}, height=0.55)
ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=12)
ax.set_xlabel("Coefficient (effect on future ROA)", fontsize=11)
ax.set_title("Effect of Risk Type Disclosures on Future ROA\n(OLS with sector FE, HC3 robust SE, N=305)", fontsize=13)
for i, (v, e) in enumerate(zip(vals, errs)):
    sig = "*" if abs(v)/e*1.96 > 1.96 else ""
    ax.text(v + (0.0003 if v >= 0 else -0.0003), i, sig, va="center",
            ha="left" if v >= 0 else "right", fontsize=14, color="black")
ax.text(0.98, 0.02, "* p < 0.05", transform=ax.transAxes,
        ha="right", fontsize=10, color="gray")
red_patch   = mpatches.Patch(color="#e74c3c", alpha=0.85, label="Negative effect")
green_patch = mpatches.Patch(color="#2ecc71", alpha=0.85, label="Positive effect")
ax.legend(handles=[red_patch, green_patch], loc="lower right", fontsize=10)
save(fig, "05_reg4_risk_type_coefficients.png")


# ════════════════════════════════════════════════════════════════════
# 2. Vagueness Ratio Distribution by Sector
# ════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
order = classified.groupby("sector")["vagueness_ratio"].median().sort_values(ascending=False).index
sns.boxplot(data=classified, x="sector", y="vagueness_ratio", order=order,
            palette=SECTOR_COLORS, ax=ax, width=0.5, linewidth=1.2)
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
ax.set_xlabel("")
ax.set_ylabel("Vagueness Ratio", fontsize=11)
ax.set_title("Vagueness of Risk Disclosures by Sector\n(share of vague language in new/changed risk text)", fontsize=13)
ax.axhline(classified["vagueness_ratio"].median(), color="gray",
           linestyle="--", linewidth=1, label=f"Overall median ({classified['vagueness_ratio'].median():.2f})")
ax.legend(fontsize=10)
save(fig, "06_vagueness_by_sector.png")


# ════════════════════════════════════════════════════════════════════
# 3. Risk Update Intensity Over Time — by Sector
# ════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 5))
for sector, grp in classified.groupby("sector"):
    ts = grp.groupby("year_new")["risk_update_intensity"].mean()
    ax.plot(ts.index, ts.values, marker="o", linewidth=2,
            label=sector, color=SECTOR_COLORS.get(sector))
ax.axvspan(2019.5, 2021.5, alpha=0.08, color="red", label="COVID period")
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Avg. Meaningful Risk Updates", fontsize=11)
ax.set_title("Risk Update Intensity Over Time by Sector", fontsize=13)
ax.legend(fontsize=9, loc="upper left")
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
save(fig, "07_risk_intensity_by_sector_timeseries.png")


# ════════════════════════════════════════════════════════════════════
# 4. Risk Type Stacked Bar — Year-over-Year
# ════════════════════════════════════════════════════════════════════
risk_cols = ["risk_cyber", "risk_operational", "risk_regulatory",
             "risk_supply_chain", "risk_financing", "risk_macroeconomic"]
risk_labels = ["Cyber", "Operational", "Regulatory", "Supply Chain", "Financing", "Macroeconomic"]
risk_palette = sns.color_palette("Set2", len(risk_cols))

yr_risk = classified.groupby("year_new")[risk_cols].mean()
yr_risk.columns = risk_labels

fig, ax = plt.subplots(figsize=(11, 5))
yr_risk.plot(kind="bar", stacked=True, ax=ax, color=risk_palette, width=0.7, edgecolor="white")
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Avg. Sentences per Firm-Year", fontsize=11)
ax.set_title("Composition of Risk Disclosure Changes by Year", fontsize=13)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(title="Risk Type", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
save(fig, "08_risk_type_composition_by_year.png")


# ════════════════════════════════════════════════════════════════════
# 5. Scatter: Risk Update Intensity vs Future ROA (with regression line)
# ════════════════════════════════════════════════════════════════════
sub = classified[["risk_update_intensity", "roa_t1", "sector"]].dropna()
sub = sub[sub["risk_update_intensity"] <= sub["risk_update_intensity"].quantile(0.95)]  # trim outliers

fig, ax = plt.subplots(figsize=(9, 6))
for sector, grp in sub.groupby("sector"):
    ax.scatter(grp["risk_update_intensity"], grp["roa_t1"],
               color=SECTOR_COLORS.get(sector), alpha=0.6, s=45, label=sector)

# OLS trend line
slope, intercept, r, p, _ = stats.linregress(sub["risk_update_intensity"], sub["roa_t1"])
x_line = np.linspace(sub["risk_update_intensity"].min(), sub["risk_update_intensity"].max(), 200)
ax.plot(x_line, intercept + slope * x_line, "k--", linewidth=2,
        label=f"OLS (slope={slope:.4f}, p={p:.3f})")
ax.set_xlabel("Risk Update Intensity (year t)", fontsize=11)
ax.set_ylabel("ROA (year t+1)", fontsize=11)
ax.set_title("Risk Update Intensity vs Future ROA", fontsize=13)
ax.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc="upper left")
save(fig, "09_intensity_vs_future_roa.png")


# ════════════════════════════════════════════════════════════════════
# 6. Nature of Change Breakdown — Stacked Bar by Year
# ════════════════════════════════════════════════════════════════════
import json
classified_dir = Path("outputs/classified")
records = []
for f in classified_dir.glob("*_classified.json"):
    parts = f.stem.replace("_classified", "").split("_")
    if len(parts) < 3:
        continue
    try:
        rows = json.loads(f.read_text())
        year_new = int(parts[2])
        for row in rows:
            records.append({"year_new": year_new, "nature": row.get("nature", "unknown")})
    except Exception:
        pass

if records:
    nat_df = pd.DataFrame(records)
    nat_yr = nat_df.groupby(["year_new", "nature"]).size().unstack(fill_value=0)
    nat_yr_pct = nat_yr.div(nat_yr.sum(axis=1), axis=0) * 100

    nature_colors = {"new_risk": "#e74c3c", "expanded_existing": "#f39c12", "boilerplate": "#95a5a6", "unknown": "#bdc3c7"}
    cols_present = [c for c in ["new_risk", "expanded_existing", "boilerplate", "unknown"] if c in nat_yr_pct.columns]
    colors_used = [nature_colors[c] for c in cols_present]

    fig, ax = plt.subplots(figsize=(11, 5))
    nat_yr_pct[cols_present].plot(kind="bar", stacked=True, ax=ax,
                                   color=colors_used, width=0.7, edgecolor="white")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("% of Classified Sentences", fontsize=11)
    ax.set_title("Nature of Risk Disclosure Changes Over Time\n(share of new vs expanded vs boilerplate)", fontsize=13)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title="Nature", labels=["New Risk", "Expanded Existing", "Boilerplate", "Unknown"],
              bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    save(fig, "10_nature_breakdown_by_year.png")


# ════════════════════════════════════════════════════════════════════
# 7. Summary Dashboard — 2×2 key findings
# ════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Risk Disclosure Research — Key Findings (Pilot: 50 S&P 500 Firms, 2015–2023)",
             fontsize=14, fontweight="bold", y=1.01)

# Top-left: vagueness vs future ROA
ax = axes[0, 0]
sub2 = classified[["vagueness_ratio", "roa_t1", "sector"]].dropna()
for sector, grp in sub2.groupby("sector"):
    ax.scatter(grp["vagueness_ratio"], grp["roa_t1"],
               color=SECTOR_COLORS.get(sector), alpha=0.55, s=30)
slope2, intercept2, _, p2, _ = stats.linregress(sub2["vagueness_ratio"], sub2["roa_t1"])
x2 = np.linspace(0, 1, 100)
ax.plot(x2, intercept2 + slope2 * x2, "k--", linewidth=1.8)
ax.set_xlabel("Vagueness Ratio (year t)")
ax.set_ylabel("ROA (year t+1)")
ax.set_title(f"Vagueness → Future ROA (p={p2:.3f}*)", fontweight="bold")

# Top-right: avg risk type counts (bar)
ax = axes[0, 1]
avg_risks = classified[risk_cols].mean()
avg_risks.index = risk_labels
bars = ax.bar(avg_risks.index, avg_risks.values, color=risk_palette, edgecolor="white", width=0.6)
ax.set_xticklabels(risk_labels, rotation=25, ha="right")
ax.set_ylabel("Avg. sentences/firm-year")
ax.set_title("Average Risk Type Mentions", fontweight="bold")

# Bottom-left: risk intensity time series
ax = axes[1, 0]
ts_all = classified.groupby("year_new")["risk_update_intensity"].mean()
ax.fill_between(ts_all.index, ts_all.values, alpha=0.25, color=PALETTE[0])
ax.plot(ts_all.index, ts_all.values, marker="o", color=PALETTE[0], linewidth=2)
ax.axvspan(2019.5, 2021.5, alpha=0.08, color="red")
ax.set_xlabel("Year")
ax.set_ylabel("Avg. meaningful updates")
ax.set_title("Risk Update Intensity Over Time", fontweight="bold")
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

# Bottom-right: Reg 4 coefficients (horizontal bar)
ax = axes[1, 1]
ax.barh(list(reversed(labels)), list(reversed(vals)),
        xerr=list(reversed(errs)), color=list(reversed(colors)), alpha=0.85,
        error_kw={"ecolor": "black", "capsize": 4}, height=0.5)
ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
ax.set_xlabel("Coefficient (→ future ROA)")
ax.set_title("Risk Type Effects on Future ROA*", fontweight="bold")
ax.text(0.98, 0.02, "* p < 0.05", transform=ax.transAxes,
        ha="right", fontsize=9, color="gray")

plt.tight_layout()
save(fig, "11_summary_dashboard.png")

print(f"\nAll plots saved to {PLOTS_DIR}/")
print("Files generated:")
for f in sorted(PLOTS_DIR.glob("*.png")):
    print(f"  {f.name}")
