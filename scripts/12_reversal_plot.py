"""
Script 12: CAR Reversal Visualisation
Shows the three-phase pattern: initial penalty → recovery → reversal.

Two panels:
  Left:  High vs Low intensity CAR lines with annotated phases
  Right: Spread (High − Low) as a single line — zero-crossings make
         each phase immediately clear
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

EVENT_CSV = Path("outputs/event_study_data.csv")
PLOTS_DIR = Path("outputs/analysis/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "axes.spines.top":  False,
    "axes.spines.right": False,
})

BLUE = "#1a6eb0"
GRAY = "#7f7f7f"
RED  = "#c0392b"

# ── Load & split ──────────────────────────────────────────────────────────────
ev  = pd.read_csv(EVENT_CSV, dtype={"cik": str})
ev  = ev.dropna(subset=["risk_update_intensity"])
med = ev["risk_update_intensity"].median()
low_g  = ev[ev["risk_update_intensity"] <= med]
high_g = ev[ev["risk_update_intensity"] >  med]

# ── Build per-day stats ───────────────────────────────────────────────────────
car_cols = sorted(
    [(int(c.replace("car_d+", "").replace("car_d", "")), c)
     for c in ev.columns if c.startswith("car_d")],
    key=lambda x: x[0]
)

days, hi_m, lo_m, hi_ci, lo_ci, spread, spread_ci, p_vals = [], [], [], [], [], [], [], []

for day, col in car_cols:
    h = high_g[col].dropna()
    l = low_g[col].dropna()
    if len(h) < 10 or len(l) < 10:
        continue
    days.append(day)
    hi_m.append(h.mean())
    lo_m.append(l.mean())
    hi_ci.append(h.sem() * 1.96)
    lo_ci.append(l.sem() * 1.96)
    sp = h.mean() - l.mean()
    spread.append(sp)
    # SE of difference (unpooled)
    se_diff = np.sqrt(h.sem()**2 + l.sem()**2)
    spread_ci.append(se_diff * 1.96)
    _, p = stats.ttest_ind(h, l)
    p_vals.append(p)

days      = np.array(days)
hi_m      = np.array(hi_m) * 100
lo_m      = np.array(lo_m) * 100
hi_ci     = np.array(hi_ci) * 100
lo_ci     = np.array(lo_ci) * 100
spread    = np.array(spread) * 100
spread_ci = np.array(spread_ci) * 100

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"wspace": 0.35})

fig.suptitle(
    "Three-Phase Market Reaction to Risk Disclosure Intensity Around 10-K Filing",
    fontsize=14, fontweight="bold", y=1.01
)

# ── Phase shading (shared across both panels) ─────────────────────────────────
phases = [
    (0,   3,  "#e74c3c", 0.07, "Phase 1\nInitial\nPenalty"),
    (3,  12,  "#27ae60", 0.07, "Phase 2\nRecovery"),
    (12, 30,  "#e67e22", 0.06, "Phase 3\nReversal"),
]

# ════════════════════════════════════════════════════
# LEFT — CAR lines with phase annotation
# ════════════════════════════════════════════════════
for x0, x1, col, alpha, label in phases:
    ax1.axvspan(x0, x1, alpha=alpha, color=col)

ax1.plot(days, hi_m, color=BLUE, linewidth=2.8,
         label=f"High Intensity  (n={len(high_g):,})")
ax1.fill_between(days, hi_m - hi_ci, hi_m + hi_ci, alpha=0.13, color=BLUE)

ax1.plot(days, lo_m, color=GRAY, linewidth=2.8, linestyle="--",
         label=f"Low Intensity   (n={len(low_g):,})")
ax1.fill_between(days, lo_m - lo_ci, lo_m + lo_ci, alpha=0.10, color=GRAY)

ax1.axvline(0, color="black", linewidth=1.5, linestyle=":", label="Filing date (d=0)")
ax1.axhline(0, color="#aaa", linewidth=0.8)

# Phase labels at top of chart
y_top = ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 1.2
phase_y = max(hi_m.max(), lo_m.max()) + hi_ci.max() * 0.6
for x0, x1, col, alpha, label in phases:
    ax1.text((x0 + x1) / 2, phase_y + 0.15,
             label, ha="center", va="bottom", fontsize=9.5,
             color=col, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, alpha=0.85))

ax1.set_xlabel("Trading Days Relative to 10-K Filing", fontsize=12)
ax1.set_ylabel("Cumulative Abnormal Return — CAR (%)", fontsize=12)
ax1.set_title("CAR by Disclosure Intensity\n(shaded = 95% confidence band)",
              fontsize=11, pad=10)
ax1.legend(fontsize=10, loc="lower left", framealpha=0.9)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}%"))
ax1.set_xlim(days[0] - 0.5, days[-1] + 0.5)

# ════════════════════════════════════════════════════
# RIGHT — Spread (High − Low) single line
# ════════════════════════════════════════════════════
for x0, x1, col, alpha, label in phases:
    ax2.axvspan(x0, x1, alpha=alpha, color=col)

ax2.plot(days, spread, color=BLUE, linewidth=3.0, label="Spread (High − Low CAR)")
ax2.fill_between(days, spread - spread_ci, spread + spread_ci,
                 alpha=0.18, color=BLUE, label="95% CI of spread")

ax2.axhline(0, color="black", linewidth=1.4, linestyle="--",
            label="Zero line (no difference)")
ax2.axvline(0, color="black", linewidth=1.2, linestyle=":", alpha=0.6)

# Annotate key zero-crossings and peak
# Find the day closest to peak (most positive spread)
peak_idx = np.argmax(spread[(days >= 5) & (days <= 15)]) + np.where(days >= 5)[0][0]
peak_day = days[peak_idx]
peak_val = spread[peak_idx]

ax2.annotate(
    f"Peak gap\nd+{int(peak_day)}: {peak_val:+.2f}%\np={p_vals[list(days).index(peak_day)]:.3f}†",
    xy=(peak_day, peak_val),
    xytext=(peak_day + 3, peak_val + 0.12),
    fontsize=10, color=BLUE,
    arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4),
    bbox=dict(boxstyle="round,pad=0.35", fc="#eef4ff", ec=BLUE, alpha=0.95)
)

# Find reversal crossings (spread goes from positive to negative)
for i in range(len(spread) - 1):
    if spread[i] > 0 and spread[i+1] <= 0 and days[i] > 5:
        ax2.axvline(days[i], color=RED, linewidth=1.4,
                    linestyle="--", alpha=0.7)
        ax2.text(days[i] + 0.4, spread.min() + 0.05,
                 f"Reversal\nd+{int(days[i])}",
                 color=RED, fontsize=9.5, fontweight="bold",
                 va="bottom")
        break

# Phase labels
spread_top = max(spread) + spread_ci.max() * 0.5
for x0, x1, col, alpha, label in phases:
    ax2.text((x0 + x1) / 2, spread_top + 0.08,
             label, ha="center", va="bottom", fontsize=9.5,
             color=col, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, alpha=0.85))

ax2.set_xlabel("Trading Days Relative to 10-K Filing", fontsize=12)
ax2.set_ylabel("CAR Spread: High − Low Intensity (%)", fontsize=12)
ax2.set_title("The Spread Between Groups\n(above zero = High outperforms Low)",
              fontsize=11, pad=10)
ax2.legend(fontsize=10, loc="lower left", framealpha=0.9)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}%"))
ax2.set_xlim(days[0] - 0.5, days[-1] + 0.5)

# ── Explanation box ───────────────────────────────────────────────────────────
explanation = (
    "Phase 1 (d0–3):  Market penalises new risk language\n"
    "                 (algorithms treat volume as a bad signal)\n\n"
    "Phase 2 (d3–12): Institutional investors re-read the 10-K,\n"
    "                 recognise disclosure as management awareness,\n"
    "                 buy the dip → spread peaks ~d+10\n\n"
    "Phase 3 (d12+):  Institutional buying exhausts; actual risks\n"
    "                 disclosed begin to be priced in → reversal"
)
fig.text(0.5, -0.08, explanation,
         ha="center", va="top", fontsize=9.5,
         family="monospace",
         bbox=dict(boxstyle="round,pad=0.7", fc="#f9f9f9",
                   ec="#cccccc", alpha=0.95))

plt.tight_layout()
fig.savefig(PLOTS_DIR / "E_car_reversal_three_phases.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved E_car_reversal_three_phases.png")
print(f"  → {PLOTS_DIR / 'E_car_reversal_three_phases.png'}")
