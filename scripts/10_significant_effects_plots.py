"""
Script 10: Significant Effects — Publication-Quality Visualisations
Generates 5 focused charts for the statistically significant findings.

Significant effects (from regression_results.txt & event study):
  • risk_cyber       → roa_t1   coef=+0.0037, p=0.002  **  (Reg 4)
  • risk_supply_chain → roa_t1  coef=+0.0040, p=0.040  *   (Reg 4)
  • vagueness_ratio  → roa_t1   coef=−0.0230, p=0.061  †   (Reg 1, marginal)
  • log_assets       → roa_t1   coef=−0.0027, p=0.027  *   (Reg 1)
  • CAR d+10 (intensity split)   t-stat ~1.76, p≈0.079 †   (Event study)

Outputs:
  outputs/analysis/plots/19_reg4_corrected_coefficients.png
  outputs/analysis/plots/20_cyber_risk_vs_roa.png
  outputs/analysis/plots/21_supply_chain_vs_roa.png
  outputs/analysis/plots/22_vagueness_vs_roa.png
  outputs/analysis/plots/23_event_study_car_d10.png
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

PANEL_CSV      = Path("outputs/final_panel.csv")
EVENT_CSV      = Path("outputs/event_study_data.csv")
PLOTS_DIR      = Path("outputs/analysis/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE = sns.color_palette("tab10")
sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

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

def save(fig, name):
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


# ── Load data ─────────────────────────────────────────────────────────────────
df         = pd.read_csv(PANEL_CSV, dtype={"cik": str})
classified = df[df["n_classified"] > 0].copy()
events     = pd.read_csv(EVENT_CSV, dtype={"cik": str})


# ════════════════════════════════════════════════════════════════════════
# 1. Corrected Reg 4 Coefficients — actual values from regression output
# ════════════════════════════════════════════════════════════════════════
# Values from regression_results.txt Reg 4 (HC3 robust SE)
coefs = {
    "Cyber Risk":        ( 0.0037, 0.00121,  True,  "p=0.002 **"),
    "Supply Chain Risk": ( 0.0040, 0.00195,  True,  "p=0.040 *"),
    "Operational Risk":  ( 0.0007, 0.000475, False, "p=0.141"),
    "Financing Risk":    (-0.0007, 0.000759, False, "p=0.357"),
    "Regulatory Risk":   (-0.0002, 0.000467, False, "p=0.669"),
}
labels  = list(coefs.keys())
vals    = [v[0] for v in coefs.values()]
ses     = [v[1] for v in coefs.values()]
sig     = [v[2] for v in coefs.values()]
p_labs  = [v[3] for v in coefs.values()]
errs    = [s * 1.96 for s in ses]    # 95% CI

# Colors: saturated for significant, muted for not
bar_colors = ["#1a6eb0" if s else "#aec7e8" for s in sig]

fig, ax = plt.subplots(figsize=(9, 5.5))
y_pos = list(range(len(labels)))

bars = ax.barh(y_pos, vals, xerr=errs, color=bar_colors,
               error_kw={"ecolor": "#555555", "capsize": 5, "linewidth": 1.6},
               height=0.55, zorder=3)
ax.axvline(0, color="black", linewidth=1.2, linestyle="--", zorder=2)

# Annotate p-values
for i, (v, e, lab, s) in enumerate(zip(vals, errs, p_labs, sig)):
    offset = 0.00025 if v >= 0 else -0.00025
    ax.text(v + offset + (e if v >= 0 else -e), i, f"  {lab}",
            va="center", ha="left" if v >= 0 else "right",
            fontsize=10, color="#333333", fontweight="bold" if s else "normal")

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=12)
ax.set_xlabel("OLS Coefficient (effect on Future ROA)", fontsize=11)
ax.set_title("Which Risk Disclosure Types Predict Future ROA?\n"
             "Reg 4: Future ROA ~ Risk Type Counts  |  HC3 Robust SE  |  N=2,418",
             fontsize=12, pad=12)

sig_patch   = mpatches.Patch(color="#1a6eb0", label="Statistically significant (p < 0.05)")
insig_patch = mpatches.Patch(color="#aec7e8", label="Not significant")
ax.legend(handles=[sig_patch, insig_patch], loc="lower right", fontsize=10,
          framealpha=0.9)
ax.set_xlim(left=min(vals) - max(errs) * 3.5)
ax.grid(axis="x", alpha=0.4)

save(fig, "19_reg4_corrected_coefficients.png")


# ════════════════════════════════════════════════════════════════════════
# 2. Cyber Risk Disclosure Level → Future ROA
# ════════════════════════════════════════════════════════════════════════
sub_cyber = classified[["risk_cyber", "roa_t1", "sector"]].dropna()

# Bin into 3 groups: None (0), Low (1–2), High (3+)
sub_cyber["cyber_group"] = pd.cut(
    sub_cyber["risk_cyber"],
    bins=[-0.1, 0, 2, 100],
    labels=["None\n(0 sentences)", "Low\n(1–2 sentences)", "High\n(3+ sentences)"]
)

group_order = ["None\n(0 sentences)", "Low\n(1–2 sentences)", "High\n(3+ sentences)"]
group_colors = ["#d0e8f7", "#6baed6", "#1a6eb0"]

fig, ax = plt.subplots(figsize=(9, 6))

# Box plots
bp = sns.boxplot(data=sub_cyber, x="cyber_group", y="roa_t1",
                 order=group_order, palette=group_colors,
                 width=0.45, linewidth=1.4, fliersize=3,
                 flierprops={"alpha": 0.4}, ax=ax)

# Overlay mean markers
means = sub_cyber.groupby("cyber_group", observed=True)["roa_t1"].mean()
ns    = sub_cyber.groupby("cyber_group", observed=True)["roa_t1"].count()
for i, g in enumerate(group_order):
    if g in means.index:
        ax.plot(i, means[g], marker="D", color="white",
                markeredgecolor="#333", markersize=8, zorder=5)

# Significance brackets
pairs = [
    (0, 1, sub_cyber[sub_cyber["cyber_group"] == group_order[0]]["roa_t1"],
          sub_cyber[sub_cyber["cyber_group"] == group_order[1]]["roa_t1"]),
    (0, 2, sub_cyber[sub_cyber["cyber_group"] == group_order[0]]["roa_t1"],
          sub_cyber[sub_cyber["cyber_group"] == group_order[2]]["roa_t1"]),
    (1, 2, sub_cyber[sub_cyber["cyber_group"] == group_order[1]]["roa_t1"],
          sub_cyber[sub_cyber["cyber_group"] == group_order[2]]["roa_t1"]),
]
y_max = sub_cyber["roa_t1"].quantile(0.95)
bar_heights = [y_max + 0.018, y_max + 0.030, y_max + 0.042]

for (x1, x2, g1, g2), yh in zip(pairs, bar_heights):
    _, p = stats.ttest_ind(g1.dropna(), g2.dropna())
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else f"p={p:.2f}"
    ax.plot([x1, x1, x2, x2], [yh - 0.006, yh, yh, yh - 0.006],
            lw=1.2, color="#444")
    ax.text((x1 + x2) / 2, yh + 0.001, stars, ha="center", va="bottom",
            fontsize=11, color="#444")

# N labels
for i, g in enumerate(group_order):
    if g in ns.index:
        ax.text(i, sub_cyber["roa_t1"].quantile(0.02) - 0.008, f"n={ns[g]:,}",
                ha="center", fontsize=9, color="#666")

ax.set_xlabel("Cyber Risk Disclosure Intensity", fontsize=12)
ax.set_ylabel("Future ROA (year t+1)", fontsize=12)
ax.set_title("Firms That Disclose Cyber Risks Have Higher Future ROA\n"
             "Cyber Risk count (year t)  →  ROA (year t+1)  |  coef=+0.0037, p=0.002**",
             fontsize=12, pad=10)
ax.set_ylim(sub_cyber["roa_t1"].quantile(0.01) - 0.015,
            y_max + 0.055)

# Add annotation text box
ax.text(0.97, 0.97,
        "Consistent with transparency signalling:\nfirms identifying cyber risk\nare better prepared, outperform peers",
        transform=ax.transAxes, fontsize=9.5, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.5", fc="#f0f7ff", ec="#1a6eb0", alpha=0.9))

save(fig, "20_cyber_risk_vs_roa.png")


# ════════════════════════════════════════════════════════════════════════
# 3. Supply Chain Risk Disclosure Level → Future ROA
# ════════════════════════════════════════════════════════════════════════
sub_sc = classified[["risk_supply_chain", "roa_t1", "sector"]].dropna()

sub_sc["sc_group"] = pd.cut(
    sub_sc["risk_supply_chain"],
    bins=[-0.1, 0, 1, 100],
    labels=["None\n(0 sentences)", "Low\n(1 sentence)", "High\n(2+ sentences)"]
)
sc_group_order  = ["None\n(0 sentences)", "Low\n(1 sentence)", "High\n(2+ sentences)"]
sc_group_colors = ["#e8f4e8", "#74c476", "#238b45"]

fig, ax = plt.subplots(figsize=(9, 6))

sns.boxplot(data=sub_sc, x="sc_group", y="roa_t1",
            order=sc_group_order, palette=sc_group_colors,
            width=0.45, linewidth=1.4, fliersize=3,
            flierprops={"alpha": 0.4}, ax=ax)

# Mean markers
sc_means = sub_sc.groupby("sc_group", observed=True)["roa_t1"].mean()
sc_ns    = sub_sc.groupby("sc_group", observed=True)["roa_t1"].count()
for i, g in enumerate(sc_group_order):
    if g in sc_means.index:
        ax.plot(i, sc_means[g], marker="D", color="white",
                markeredgecolor="#333", markersize=8, zorder=5)

# Significance bracket (None vs High — most interesting)
g_none = sub_sc[sub_sc["sc_group"] == sc_group_order[0]]["roa_t1"].dropna()
g_high = sub_sc[sub_sc["sc_group"] == sc_group_order[2]]["roa_t1"].dropna()
_, p_sc = stats.ttest_ind(g_none, g_high)
yh_sc = sub_sc["roa_t1"].quantile(0.95) + 0.022
stars_sc = "***" if p_sc < 0.001 else "**" if p_sc < 0.01 else "*" if p_sc < 0.05 else f"p={p_sc:.2f}"
ax.plot([0, 0, 2, 2], [yh_sc - 0.006, yh_sc, yh_sc, yh_sc - 0.006],
        lw=1.2, color="#444")
ax.text(1, yh_sc + 0.001, stars_sc, ha="center", va="bottom", fontsize=11, color="#444")

# None vs Low
g_low = sub_sc[sub_sc["sc_group"] == sc_group_order[1]]["roa_t1"].dropna()
_, p_nl = stats.ttest_ind(g_none, g_low)
yh_nl = sub_sc["roa_t1"].quantile(0.95) + 0.008
stars_nl = "***" if p_nl < 0.001 else "**" if p_nl < 0.01 else "*" if p_nl < 0.05 else f"p={p_nl:.2f}"
ax.plot([0, 0, 1, 1], [yh_nl - 0.004, yh_nl, yh_nl, yh_nl - 0.004],
        lw=1.2, color="#444")
ax.text(0.5, yh_nl + 0.001, stars_nl, ha="center", va="bottom", fontsize=11, color="#444")

# N labels
for i, g in enumerate(sc_group_order):
    if g in sc_ns.index:
        ax.text(i, sub_sc["roa_t1"].quantile(0.02) - 0.008, f"n={sc_ns[g]:,}",
                ha="center", fontsize=9, color="#666")

ax.set_xlabel("Supply Chain Risk Disclosure Intensity", fontsize=12)
ax.set_ylabel("Future ROA (year t+1)", fontsize=12)
ax.set_title("Supply Chain Risk Disclosers Outperform Peers\n"
             "Supply Chain count (year t)  →  ROA (year t+1)  |  coef=+0.0040, p=0.040*",
             fontsize=12, pad=10)
ax.set_ylim(sub_sc["roa_t1"].quantile(0.01) - 0.015,
            sub_sc["roa_t1"].quantile(0.95) + 0.060)

ax.text(0.97, 0.97,
        "Proactive supply chain disclosures signal\noperational awareness — these firms\ntend to manage disruptions more effectively",
        transform=ax.transAxes, fontsize=9.5, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.5", fc="#f0fff0", ec="#238b45", alpha=0.9))

save(fig, "21_supply_chain_vs_roa.png")


# ════════════════════════════════════════════════════════════════════════
# 4. Vagueness Ratio → Future ROA  (scatter + OLS + CI band)
# ════════════════════════════════════════════════════════════════════════
sub_v = classified[["vagueness_ratio", "roa_t1", "sector"]].dropna()
# Remove top 1% ROA outliers for cleaner plot
sub_v = sub_v[sub_v["roa_t1"] <= sub_v["roa_t1"].quantile(0.99)]
sub_v = sub_v[sub_v["roa_t1"] >= sub_v["roa_t1"].quantile(0.01)]
# Focus on range where vagueness < 1.0 (most interesting variation)
sub_v_var = sub_v[sub_v["vagueness_ratio"] < 1.0]

fig, ax = plt.subplots(figsize=(10, 6.5))

# Scatter by sector
for sector, grp in sub_v.groupby("sector"):
    ax.scatter(grp["vagueness_ratio"], grp["roa_t1"],
               color=SECTOR_COLORS.get(sector, "#aaaaaa"),
               alpha=0.35, s=28, label=sector, zorder=2)

# OLS trend line with 95% confidence interval
slope, intercept, r_val, p_val, se = stats.linregress(sub_v["vagueness_ratio"], sub_v["roa_t1"])
x_line = np.linspace(sub_v["vagueness_ratio"].min(), 1.0, 300)
y_line = intercept + slope * x_line
n = len(sub_v)
x_mean = sub_v["vagueness_ratio"].mean()
sx = sub_v["vagueness_ratio"].std()
# SE of fitted values
se_fit = se * np.sqrt(1/n + (x_line - x_mean)**2 / ((n-1) * sx**2))
t_crit = stats.t.ppf(0.975, df=n-2)

ax.plot(x_line, y_line, color="#c0392b", linewidth=2.5, zorder=4,
        label=f"OLS: slope={slope:.4f}  p={p_val:.3f}†")
ax.fill_between(x_line,
                y_line - t_crit * se_fit,
                y_line + t_crit * se_fit,
                color="#e74c3c", alpha=0.12, zorder=3, label="95% CI band")

# Sector legend (right side)
handles, lbls = ax.get_legend_handles_labels()
# Separate OLS line and CI from sector patches
line_handles = [h for h, l in zip(handles, lbls) if "OLS" in l or "CI" in l]
line_labels  = [l for l in lbls if "OLS" in l or "CI" in l]
sector_handles = [h for h, l in zip(handles, lbls) if "OLS" not in l and "CI" not in l]
sector_labels  = [l for l in lbls if "OLS" not in l and "CI" not in l]

ax.legend(line_handles, line_labels, loc="upper right", fontsize=10, framealpha=0.9)
sec_leg = ax.legend(sector_handles, sector_labels,
                    loc="lower left", fontsize=8.5, ncol=2,
                    title="Sector", title_fontsize=9, framealpha=0.9)
ax.add_artist(sec_leg)
# Re-add the OLS legend (it was replaced above)
ax.legend(line_handles, line_labels, loc="upper right", fontsize=10, framealpha=0.9)

# Annotation
ax.text(0.41, ax.get_ylim()[1] * 0.92,
        f"R = {r_val:.3f}\np = {p_val:.3f}†\nn = {n:,}",
        fontsize=10.5, va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#c0392b", alpha=0.9))

ax.set_xlabel("Vagueness Ratio (year t) — share of vague language in new/changed risk text", fontsize=11)
ax.set_ylabel("Future ROA (year t+1)", fontsize=11)
ax.set_title("Higher Risk Disclosure Vagueness Predicts Lower Future Profitability\n"
             "Vagueness Ratio (year t)  →  ROA (year t+1)  |  coef=−0.0230, p=0.061†",
             fontsize=12, pad=12)
ax.axvline(sub_v["vagueness_ratio"].median(), color="gray", linewidth=1,
           linestyle=":", alpha=0.8, label=f"Median vagueness ({sub_v['vagueness_ratio'].median():.2f})")

save(fig, "22_vagueness_vs_roa.png")


# ════════════════════════════════════════════════════════════════════════
# 5. Event Study: CAR at d+10 by Risk Update Intensity (tercile)
# ════════════════════════════════════════════════════════════════════════
ev = events.copy()
car_cols = [c for c in ev.columns if c.startswith("car_d")]

# Build CAR path
car_days = []
for c in car_cols:
    try:
        d = int(c.replace("car_d", "").replace("+", ""))
        car_days.append((d, c))
    except ValueError:
        pass
car_days.sort(key=lambda x: x[0])

# Intensity tercile split
ev = ev.dropna(subset=["risk_update_intensity"])
try:
    ev["intensity_group"] = pd.qcut(ev["risk_update_intensity"], q=3,
                                     labels=["Low Intensity", "Medium Intensity", "High Intensity"],
                                     duplicates="drop")
except ValueError:
    ranks = ev["risk_update_intensity"].rank(method="first", na_option="bottom")
    ev["intensity_group"] = pd.cut(ranks, bins=3,
                                    labels=["Low Intensity", "Medium Intensity", "High Intensity"])

group_labels = ["Low Intensity", "Medium Intensity", "High Intensity"]
int_colors   = ["#fdae6b", "#f16913", "#7f2704"]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Event Study: Risk Update Intensity and Abnormal Returns Around 10-K Filing\n"
             "(Market Model CAR, Estimation Window: −120 to −10 days)",
             fontsize=13, fontweight="bold", y=1.01)

# ── Left panel: full drift path ───────────────────────────────────────────────
ax = axes[0]
for grp_label, color in zip(group_labels, int_colors):
    sub_g = ev[ev["intensity_group"] == grp_label]
    means = []
    ci_lo = []
    ci_hi = []
    for day, col in car_days:
        if col not in sub_g.columns:
            continue
        vals_g = sub_g[col].dropna()
        if len(vals_g) < 5:
            continue
        m = vals_g.mean()
        se_g = vals_g.sem()
        means.append((day, m))
        ci_lo.append((day, m - 1.96 * se_g))
        ci_hi.append((day, m + 1.96 * se_g))
    if means:
        days_x  = [x[0] for x in means]
        means_y = [x[1] for x in means]
        lo_y    = [x[1] for x in ci_lo]
        hi_y    = [x[1] for x in ci_hi]
        n_grp   = len(ev[ev["intensity_group"] == grp_label])
        ax.plot(days_x, means_y, linewidth=2.2, color=color,
                label=f"{grp_label} (n={n_grp:,})")
        ax.fill_between(days_x, lo_y, hi_y, alpha=0.12, color=color)

ax.axvline(0, color="black", linewidth=1.5, linestyle="--", label="Filing date")
ax.axhline(0, color="#888", linewidth=0.8)
ax.axvspan(10, 12, alpha=0.12, color="#1a6eb0", label="d+10 window")
ax.set_xlabel("Trading Days Relative to 10-K Filing", fontsize=11)
ax.set_ylabel("Cumulative Abnormal Return (CAR)", fontsize=11)
ax.set_title("CAR Drift by Intensity (−5 to +30 days)", fontsize=11)
ax.legend(fontsize=9.5, loc="lower left")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))

# ── Right panel: bar chart at specific milestones with t-test vs Low ──────────
ax2 = axes[1]
milestones = [0, 5, 10, 20, 30]
x = np.arange(len(milestones))
width = 0.25

for i_g, (grp_label, color) in enumerate(zip(group_labels, int_colors)):
    sub_g  = ev[ev["intensity_group"] == grp_label]
    vals_m = []
    err_m  = []
    for day in milestones:
        col = f"car_d+{day}" if day >= 0 else f"car_d{day}"
        if col in sub_g.columns:
            v = sub_g[col].dropna()
            vals_m.append(v.mean())
            err_m.append(v.sem() * 1.96)
        else:
            vals_m.append(np.nan)
            err_m.append(0)
    bars = ax2.bar(x + i_g * width, vals_m, width=width, color=color,
                   alpha=0.85, label=grp_label, edgecolor="white",
                   yerr=err_m, error_kw={"ecolor": "#333", "capsize": 3, "linewidth": 1})

# Annotate d+10 with t-test p-value (High vs Low)
low_d10  = ev[ev["intensity_group"] == "Low Intensity"]["car_d+10"].dropna()
high_d10 = ev[ev["intensity_group"] == "High Intensity"]["car_d+10"].dropna()
_, p_d10 = stats.ttest_ind(high_d10, low_d10)
d10_idx = milestones.index(10)
y_annot = max(high_d10.mean(), low_d10.mean()) + high_d10.sem() * 2 + 0.003
ax2.annotate(f"High vs Low\np={p_d10:.3f}†",
             xy=(d10_idx + width, y_annot),
             xytext=(d10_idx + width, y_annot + 0.005),
             ha="center", fontsize=9.5, color="#1a6eb0",
             bbox=dict(boxstyle="round,pad=0.3", fc="#eef4ff", ec="#1a6eb0", alpha=0.9))

ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_xticks(x + width)
ax2.set_xticklabels([f"d+{d}" if d >= 0 else f"d{d}" for d in milestones])
ax2.set_xlabel("Event Day", fontsize=11)
ax2.set_ylabel("Mean CAR", fontsize=11)
ax2.set_title("Mean CAR at Key Milestones by Intensity Group", fontsize=11)
ax2.legend(fontsize=9.5)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))

plt.tight_layout()
save(fig, "23_event_study_car_d10.png")


print(f"\nAll 5 significant-effects plots saved to {PLOTS_DIR}/")
print("Files generated:")
for f in sorted(PLOTS_DIR.glob("1[9-9]_*.png")) :
    print(f"  {f.name}")
for f in sorted(PLOTS_DIR.glob("2[0-9]_*.png")):
    print(f"  {f.name}")
