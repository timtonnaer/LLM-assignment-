# Do Firms Foreshadow Negative Outcomes Through Changes in Risk Disclosure?

Research project analyzing how S&P 500 firms change their 10-K risk factor disclosures before negative economic outcomes.

## Prerequisites

- Python 3.11+
- [WRDS account](https://wrds-www.wharton.upenn.edu/) (for Compustat + CRSP data)
- [Anthropic API key](https://console.anthropic.com/) with credits (~$1–2 for full S&P 500)
- Large data files shared separately (see below)

## Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd risk_project

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate      # Mac/Linux
# venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-...
# Add this to ~/.zshrc to make it permanent
```

## Large Data Files (shared separately)

The following directories are **not in git** (too large). Get them from the shared Google Drive / institutional server:

| Directory | Size | Contents |
|---|---|---|
| `data/extracted/` | ~65 GB | Raw SEC EDGAR 10-K filings (2011–2025) |
| `data/risk_factors/` | ~6 GB | Extracted Risk Factors sections |
| `outputs/diffs/` | ~1.8 GB | Year-over-year sentence diff JSONs |

Everything in `outputs/*.csv` and `outputs/classified/` **is** in the repo — you can run scripts 04–08 without re-doing the expensive steps.

## Pipeline

Run scripts in order. Each step is **idempotent** (safe to re-run).

```bash
# Step 1: Extract the 50 pilot firms from the Excel file
python scripts/01_extract_pilot_firms.py

# Step 2: Index which diff files exist for pilot firms (2015–2023)
python scripts/02_filter_diffs.py

# Step 3: LLM classification of risk disclosure changes
# Skips already-classified diffs automatically
python scripts/03_llm_classify.py --backend claude
# Or with local Ollama (free, but slower):
# python scripts/03_llm_classify.py --backend ollama --model llama3.2

# Step 4: Build firm-year variable panel
python scripts/04_construct_variables.py

# Step 5: Pull Compustat + CRSP data from WRDS
# Will prompt for WRDS credentials on first run
python scripts/05_fetch_wrds.py

# Step 6: Merge panel, run regressions, generate plots
python scripts/06_merge_and_analyze.py

# Step 7: Additional visualizations
python scripts/07_visualize_results.py

# Step 8: Event study (±3 day CAR around 10-K filing dates)
python scripts/08_event_study.py
```

## Key Outputs

| File | Description |
|---|---|
| `outputs/pilot_firms.csv` | 50 pilot firms with CIKs and sectors |
| `outputs/pilot_diff_index.csv` | Index of available diffs per firm-year |
| `outputs/classified/` | LLM classification JSONs (one per diff) |
| `outputs/firm_year_variables.csv` | Key text variables at firm-year level |
| `outputs/compustat_panel.csv` | Compustat financials (ROA, leverage, etc.) |
| `outputs/crsp_returns.csv` | CRSP annual stock returns |
| `outputs/final_panel.csv` | Merged panel — ready for analysis |
| `outputs/event_study_data.csv` | CAR data for ±3 day event window |
| `outputs/analysis/regression_results.txt` | OLS regression tables |
| `outputs/analysis/plots/` | All visualizations (11 PNG files) |

## Key Findings (Pilot — 50 firms, 2015–2023)

1. **Vague risk language predicts higher future ROA** (p=0.018) — concrete disclosures signal genuine trouble
2. **Financing & regulatory risk disclosures predict lower future ROA** (p<0.05) — real foreshadowing signal
3. **Markets don't react to risk update quantity at filing time** (event study, p=0.935) — signal is underpriced
4. **COVID caused a 3× spike in Health Care and IT risk updates** in 2020–2021
5. **~85% of risk changes are boilerplate** — LLM filtering is essential to isolate signal

## Adding More Companies

1. Add firms to `outputs/pilot_firms.csv` (or update the Excel pilot sheet)
2. Re-run scripts 02 → 03 → 04 → 06
   - Script 03 automatically skips already-classified diffs
   - Cost: ~$0.001 per company per year with Claude Haiku + prompt caching

## Dev Servers

```bash
# JupyterLab (interactive analysis at localhost:8888)
venv/bin/jupyter lab --no-browser --port=8888 --NotebookApp.token=''

# Results dashboard (localhost:8050)
python scripts/dashboard.py
```
