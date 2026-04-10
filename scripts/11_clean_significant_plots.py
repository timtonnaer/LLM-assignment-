"""
Script 11: Clean Significant Effects — Simple, Intuitive Visualisations
4 plots, each showing one clear comparison with minimal clutter.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

PANEL_CSV  = Path("outputs/final_panel.csv")
EVENT_CSV  = Path("outputs/event_study_data.csv")
PLOTS_DIR  = Path("outputs/analysis/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
})

BLUE  = "#1a6eb0"
GREEN = "#238b45"
RED   = "#c0392b"
GRAY  = "#95a5a6"

df         = pd.read_csv(PANEL_CSV, dtype={"cik": str})
classified = df[df["n_classified"] > 0].dropna(subset=["roa_t1"]).copy()
events     = pd.read_csv(EVENT_CSV, dtype={"cik": str})


def bar_comparison(ax, val_a, val_b, label_a, label_b,
                   color_a, color_b, ylabel, title, subtitle,
                   p_val, coef_str):
    """Draw a clean two-bar comparison with CI, n-labels and annotation."""
    means = [val_a.mean(), val_b.mean()]
    cis   = [val_a.sem() * 1.96, val_b.sem() * 1.96]
    ns    = [len(val_a), len(val_b)]
    colors = [color_a, color_b]
    labels = [label_a, label_b]

    bars = ax.bar(labels, means, color=colors, width=0.45,
                  yerr=cis, error_kw={"ecolor": "#333", "capsize": 8,
                                       "linewidth": 1.8, "elinewidth": 1.8},
                  alpha=0.88, edgecolor="white", linewidth=0, zorder=3)

    # Value labels on bars
    for bar, m, ci, n in zip(bars, means, cis, ns):
        ax.text(bar.get_x() + bar.get_width() / 2,
                m + ci + 0.001,
                f"{m:.3f}", ha="center", va="bottom",
                fontsize=13, fontweight="bold", color="#222")
        ax.text(bar.get_x() + bar.get_width() / 2,
                0.002, f"n = {n:,}", ha="center", va="bottom",
                fontsize=10, color="white", fontweight="bold")

    # Significance bracket
    y_bracket = max(means) + max(cis) + 0.012
    ax.plot([0, 0, 1, 1],
            [y_bracket - 0.005, y_bracket, y_bracket, y_bracket - 0.005],
            lw=1.4, color="#444")
    stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else \
            "*"   if p_val < 0.05  else "†"  if p_val < 0.10 else f"p={p_val:.2f}"
    ax.text(0.5, y_bracket + 0.001, stars, ha="center", va="bottom",
            fontsize=15, color="#222")

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{title}\n{subtitle}", fontsize=13, fontweight="bold",
                 pad=12, loc="center")
    ax.set_ylim(0, y_bracket + 0.025)
    ax.tick_params(axis="x", labelsize=12)

    # Bottom annotation
    ax.text(0.98, 0.03,
            f"OLS coef {coef_str}  |  p = {p_val:.3f}",
            transform=ax.transAxes, fontsize=9.5,
            ha="right", color="#555",
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec="#bbb", alpha=0.9))


# ══════════════════════════════════════════════════════════════════════
# FIGURE 1 — Cyber Risk Disclosers vs Non-Disclosers
# ══════════════════════════════════════════════════════════════════════
no_cyber  = classified[classified["risk_cyber"] == 0]["roa_t1"].dropna()
yes_cyber = classified[classified["risk_cyber"] >  0]["roa_t1"].dropna()
_, p_cyber = stats.ttest_ind(yes_cyber, no_cyber)

fig, ax = plt.subplots(figsize=(7, 6))
bar_comparison(
    ax,
    val_a=no_cyber,   label_a="Did NOT disclose\ncyber risk",   color_a=GRAY,
    val_b=yes_cyber,  label_b="DID disclose\ncyber risk",       color_b=BLUE,
    ylabel="Mean Future ROA (year t+1)",
    title="Firms that disclose cyber risk\nearn higher future profits",
    subtitle="Cyber Risk Sentences in 10-K  →  ROA one year later",
    p_val=p_cyber,
    coef_str="= +0.0037"
)
fig.savefig(PLOTS_DIR / "A_cyber_disclosers_vs_not.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved A_cyber_disclosers_vs_not.png")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 2 — Supply Chain Disclosers vs Non-Disclosers
# ══════════════════════════════════════════════════════════════════════
no_sc  = classified[classified["risk_supply_chain"] == 0]["roa_t1"].dropna()
yes_sc = classified[classified["risk_supply_chain"] >  0]["roa_t1"].dropna()
_, p_sc = stats.ttest_ind(yes_sc, no_sc)

fig, ax = plt.subplots(figsize=(7, 6))
bar_comparison(
    ax,
    val_a=no_sc,   label_a="Did NOT disclose\nsupply chain risk",  color_a=GRAY,
    val_b=yes_sc,  label_b="DID disclose\nsupply chain risk",      color_b=GREEN,
    ylabel="Mean Future ROA (year t+1)",
    title="Firms that disclose supply chain risk\nearn higher future profits",
    subtitle="Supply Chain Risk Sentences in 10-K  →  ROA one year later",
    p_val=p_sc,
    coef_str="= +0.0040"
)
fig.savefig(PLOTS_DIR / "B_supply_chain_disclosers_vs_not.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved B_supply_chain_disclosers_vs_not.png")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 3 — Concrete vs Vague Language → Future ROA
# ══════════════════════════════════════════════════════════════════════
sub_v = classified.dropna(subset=["vagueness_ratio", "roa_t1"])
# Split at median vagueness
med_v = sub_v["vagueness_ratio"].median()
concrete = sub_v[sub_v["vagueness_ratio"] <= med_v]["roa_t1"]
vague    = sub_v[sub_v["vagueness_ratio"] >  med_v]["roa_t1"]
_, p_vag = stats.ttest_ind(concrete, vague)

fig, ax = plt.subplots(figsize=(7, 6))
bar_comparison(
    ax,
    val_a=concrete, label_a="Concrete language\n(below median vagueness)", color_a=BLUE,
    val_b=vague,    label_b="Vague language\n(above median vagueness)",    color_b=RED,
    ylabel="Mean Future ROA (year t+1)",
    title="Firms using vaguer risk language\nhave lower future profitability",
    subtitle="Vagueness Ratio in 10-K  →  ROA one year later",
    p_val=p_vag,
    coef_str="= −0.0230"
)
fig.savefig(PLOTS_DIR / "C_vague_vs_concrete_roa.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved C_vague_vs_concrete_roa.png")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 4 — Event Study: High vs Low Intensity (2 clean lines)
# ══════════════════════════════════════════════════════════════════════
ev = events.dropna(subset=["risk_update_intensity"]).copy()

# Split at median into just two groups for clarity
med_int = ev["risk_update_intensity"].median()
low_g   = ev[ev["risk_update_intensity"] <= med_int]
high_g  = ev[ev["risk_update_intensity"] >  med_int]

car_cols = sorted(
    [(int(c.replace("car_d+", "").replace("car_d", "")), c)
     for c in ev.columns if c.startswith("car_d")],
    key=lambda x: x[0]
)

days, low_means, high_means = [], [], []
low_ci, high_ci = [], []
for day, col in car_cols:
    l = low_g[col].dropna()
    h = high_g[col].dropna()
    if len(l) < 10 or len(h) < 10:
        continue
    days.append(day)
    low_means.append(l.mean())
    high_means.append(h.mean())
    low_ci.append(l.sem() * 1.96)
    high_ci.append(h.sem() * 1.96)

days       = np.array(days)
low_means  = np.array(low_means)
high_means = np.array(high_means)
low_ci     = np.array(low_ci)
high_ci    = np.array(high_ci)

# t-test at d+10
_, p_d10 = stats.ttest_ind(
    high_g["car_d+10"].dropna(),
    low_g["car_d+10"].dropna()
)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(days, high_means * 100, color=BLUE, linewidth=2.8,
        label=f"High Intensity (n={len(high_g):,}) — many new risk disclosures")
ax.fill_between(days,
                (high_means - high_ci) * 100,
                (high_means + high_ci) * 100,
                alpha=0.13, color=BLUE)

ax.plot(days, low_means * 100, color=GRAY, linewidth=2.8, linestyle="--",
        label=f"Low Intensity  (n={len(low_g):,}) — few or no new risk disclosures")
ax.fill_between(days,
                (low_means - low_ci) * 100,
                (low_means + low_ci) * 100,
                alpha=0.10, color=GRAY)

# Filing date line
ax.axvline(0, color="black", linewidth=1.6, linestyle="--", label="10-K filing date")
ax.axhline(0, color="#888", linewidth=0.8)

# Annotate d+10 gap
d10_idx = list(days).index(10) if 10 in days else None
if d10_idx is not None:
    gap = (high_means[d10_idx] - low_means[d10_idx]) * 100
    ax.annotate(
        f"Gap at day +10\n{gap:+.2f}%  (p={p_d10:.3f}†)",
        xy=(10, high_means[d10_idx] * 100),
        xytext=(14, high_means[d10_idx] * 100 + 0.25),
        fontsize=10.5, color=BLUE,
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4),
        bbox=dict(boxstyle="round,pad=0.4", fc="#eef4ff", ec=BLUE, alpha=0.92)
    )

ax.set_xlabel("Trading Days Relative to 10-K Filing Date", fontsize=12)
ax.set_ylabel("Cumulative Abnormal Return — CAR (%)", fontsize=12)
ax.set_title(
    "Firms with More New Risk Disclosures Face a Short-Term Market Penalty\n"
    "but Recover Within ~10 Days of Filing",
    fontsize=13, fontweight="bold", pad=12
)
ax.legend(fontsize=10.5, loc="lower left", framealpha=0.9)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}%"))

fig.savefig(PLOTS_DIR / "D_event_study_high_vs_low_intensity.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved D_event_study_high_vs_low_intensity.png")


print("\nAll 4 clean plots saved:")
for name in ["A_cyber_disclosers_vs_not.png",
             "B_supply_chain_disclosers_vs_not.png",
             "C_vague_vs_concrete_roa.png",
             "D_event_study_high_vs_low_intensity.png"]:
    print(f"  {PLOTS_DIR / name}")
