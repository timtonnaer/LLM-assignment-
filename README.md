# Risk Disclosure Change Analysis — S&P 500 10-K Filings

**Research Question:** Do firms subtly expand or modify their 10-K risk disclosures *before* negative economic outcomes — and do those textual signals predict future performance and market reactions?

**Sample:** ~460 S&P 500 firms, 2015–2023 | ~2,300 firm-year observations  
**Method:** LLM classification (Claude Haiku) of year-over-year risk factor changes → OLS regressions + event study + long-short portfolio

---

## Key Findings

| Finding | Effect | Significance |
|---------|--------|-------------|
| Cyber risk disclosure → higher future ROA | +0.37pp per sentence | p = 0.002 ** |
| Supply chain disclosure → higher future ROA | +0.40pp per sentence | p = 0.040 * |
| Vaguer language → lower future ROA | −2.3pp per unit vagueness | p = 0.061 † |
| High intensity disclosers face short-term market penalty, recover by d+10 | CAR gap | p = 0.079 † |

**Interpretation:** Firms that name specific risks (cyber, supply chain) outperform peers — consistent with a transparency/preparedness signalling story. Vague, hedged language predicts underperformance. The market initially overreacts to disclosure volume but partially corrects within 10 days, then reverses — suggesting a three-phase information processing dynamic.

---

## Project Structure

```
risk_project/
├── scripts/                             # All analysis scripts (run in order)
│   ├── 01_extract_pilot_firms.py        # Read firm list from Excel
│   ├── 02_filter_diffs.py               # Index year-over-year diff files
│   ├── 03_llm_classify.py               # LLM classification via Claude Batch API
│   ├── 03b_collect_batch.py             # Collect results from existing batch
│   ├── 03c_fix_unknowns.py              # Re-classify any "unknown" labels
│   ├── 04_construct_variables.py        # Aggregate to firm-year panel variables
│   ├── 05_fetch_wrds.py                 # Pull Compustat + CRSP from WRDS
│   ├── 06_merge_and_analyze.py          # Merge panel, run OLS regressions
│   ├── 07_visualize_results.py          # General visualisations (11 plots)
│   ├── 08_event_study.py                # Extended event study [-5, +30] window
│   ├── 09_portfolio.py                  # Long-short portfolio construction
│   ├── 10_significant_effects_plots.py  # Detailed plots for significant effects
│   ├── 11_clean_significant_plots.py    # Simple two-group comparison plots ← start here
│   └── 12_reversal_plot.py              # Three-phase CAR reversal chart
│
├── outputs/
│   ├── all_firms.csv                    # 460 firms with CIK, ticker, sector
│   ├── all_diff_index.csv               # 2,273 diff files indexed
│   ├── firm_year_variables.csv          # Text-based variables (firm × year)
│   ├── compustat_panel.csv              # WRDS Compustat financials
│   ├── crsp_returns.csv                 # WRDS CRSP annual returns
│   ├── final_panel.csv                  # Master merged panel — START HERE for analysis
│   ├── event_study_data.csv             # CAR data for all 2,341 events
│   ├── portfolio_returns.csv            # Daily L-S portfolio returns
│   ├── classified/                      # Per-firm-year LLM classification JSONs
│   └── analysis/
│       ├── regression_results.txt       # Full OLS output tables (Reg 1–4)
│       └── plots/                       # All visualisations (PNG)
│           ├── A_cyber_disclosers_vs_not.png
│           ├── B_supply_chain_disclosers_vs_not.png
│           ├── C_vague_vs_concrete_roa.png
│           ├── D_event_study_high_vs_low_intensity.png
│           └── E_car_reversal_three_phases.png   ← most interesting
│
├── requirements.txt
└── README.md
```

---

## Quickstart — Just Run the Visualisations

If you have been given the pre-processed data files (`final_panel.csv`, `event_study_data.csv`, `portfolio_returns.csv`), you do **not** need an API key or WRDS access. Just clone and plot:

```bash
git clone https://github.com/timtonnaer/LLM-assignment-.git
cd LLM-assignment-
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Drop the shared data files into outputs/ then:
python scripts/11_clean_significant_plots.py   # 4 clean two-group charts
python scripts/12_reversal_plot.py             # CAR three-phase reversal
python scripts/07_visualize_results.py         # full dashboard
```

---

## Full Pipeline (from scratch)

Only needed if you want to re-run LLM classification or extend the sample.

### Prerequisites
- **Anthropic API key** — set as `export ANTHROPIC_API_KEY=sk-ant-...`
- **WRDS account** — prompted on first run of script 05

```bash
python scripts/01_extract_pilot_firms.py --sample full
python scripts/02_filter_diffs.py        --sample full
python scripts/03_llm_classify.py        --sample full   # ~$8–10, uses Batch API
python scripts/03c_fix_unknowns.py                       # fix any failed labels
python scripts/04_construct_variables.py
python scripts/05_fetch_wrds.py                          # WRDS credentials required
python scripts/06_merge_and_analyze.py
python scripts/07_visualize_results.py
python scripts/08_event_study.py
python scripts/09_portfolio.py
python scripts/10_significant_effects_plots.py
python scripts/11_clean_significant_plots.py
python scripts/12_reversal_plot.py
```

---

## Key Columns in `final_panel.csv`

| Column | Description |
|--------|-------------|
| `cik`, `ticker`, `sector` | Firm identifiers |
| `year_new` | Filing year |
| `risk_update_intensity` | Count of genuinely new/expanded risk sentences |
| `vagueness_ratio` | Share of vague language (0 = fully concrete, 1 = fully vague) |
| `boilerplate_ratio` | Share of boilerplate sentences |
| `risk_cyber` | Count of cyber risk sentences added |
| `risk_supply_chain` | Count of supply chain risk sentences added |
| `risk_operational` | Count of operational risk sentences added |
| `risk_financing` | Count of financing risk sentences added |
| `risk_regulatory` | Count of regulatory risk sentences added |
| `risk_macroeconomic` | Count of macro risk sentences added |
| `roa`, `roa_t1` | Return on assets (current year, next year) |
| `rev_growth_t1` | Revenue growth next year |
| `annual_ret`, `annual_ret_t1` | Annual stock return (current, next year) |
| `leverage`, `log_assets` | Control variables |

---

## Ideas for Further Work

### Visualisations
- **Sector breakdown** of the cyber/supply chain effect — does it hold equally in IT vs Financials?
- **Year-by-year coefficient plot** — did the cyber effect strengthen post-2020?
- **Heatmap** of risk type co-occurrence — which risk types appear together?
- **Firm-level trajectories** — track individual firms' vagueness ratio over time

### Analysis Extensions
- **Placebo test** — do current-year variables predict *past* ROA? (should be zero)
- **Non-linear effects** — does very high cyber disclosure eventually backfire?
- **Interaction effects** — does vagueness × intensity predict outcomes together?
- **Text similarity** — use the `similarity` column to study year-on-year copy-paste behaviour
- **Pre/post COVID subsample** — do the effects differ before and after 2020?
- **Firm size split** — do small and large firms show different disclosure patterns?

### Data Extensions
- Expand to all SEC EDGAR filers (not just S&P 500)
- Add analyst forecast data — does disclosure reduce forecast dispersion?
- Add ESG scores — is there a link between risk transparency and ESG ratings?

---

## LLM Classification

Each added/changed sentence in a firm's risk section is classified by `claude-haiku-4-5-20251001`:

- **risk_type:** `operational | cyber | regulatory | supply_chain | financing | macroeconomic | other`
- **nature:** `new_risk | expanded_existing | boilerplate`
- **style:** `concrete | vague`

Boilerplate sentences are pre-filtered using 30+ regex patterns before the API call to reduce cost. Uses the Anthropic Message Batches API (50% cheaper than standard). Total cost for ~460 firms × 8 years ≈ $8–10.

---

## Large Data Files (not in git)

The following are too large for GitHub. Shared separately:

| Path | Size | Contents |
|------|------|----------|
| `outputs/diffs/` | ~1.8 GB | Year-over-year sentence diff JSONs |
| `outputs/classified/` | ~500 MB | LLM classification JSONs |
| `outputs/final_panel.csv` | ~2 MB | Master analysis panel |
| `outputs/event_study_data.csv` | ~1 MB | CAR event data |

---

## Dependencies

```
anthropic>=0.40
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scipy>=1.10
statsmodels>=0.14
wrds>=3.1
openpyxl>=3.1
```

Install: `pip install -r requirements.txt`

---

## Contact

Tim Tonnaer — project owner  
Repository: https://github.com/timtonnaer/LLM-assignment-
