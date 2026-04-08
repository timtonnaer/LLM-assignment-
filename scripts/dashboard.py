"""
Results Dashboard
Interactive visualization of risk disclosure classification results.
Run: python scripts/dashboard.py
Open: http://localhost:8050
"""

import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from dash import Dash, html, dcc, dash_table
import dash_bootstrap_components as dbc

CLASSIFIED_DIR = Path("outputs/classified")
DIFF_INDEX_CSV = Path("outputs/pilot_diff_index.csv")
PILOT_FIRMS_CSV = Path("outputs/pilot_firms.csv")
FIRM_VARS_CSV = Path("outputs/firm_year_variables.csv")
FINAL_PANEL_CSV = Path("outputs/final_panel.csv")

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])


def load_data():
    data = {}

    if PILOT_FIRMS_CSV.exists():
        data["firms"] = pd.read_csv(PILOT_FIRMS_CSV, dtype={"cik": str})

    if DIFF_INDEX_CSV.exists():
        data["index"] = pd.read_csv(DIFF_INDEX_CSV, dtype={"cik": str})

    if FIRM_VARS_CSV.exists():
        data["vars"] = pd.read_csv(FIRM_VARS_CSV, dtype={"cik": str})

    if FINAL_PANEL_CSV.exists():
        data["panel"] = pd.read_csv(FINAL_PANEL_CSV, dtype={"cik": str})

    # Count classified files
    classified_files = list(CLASSIFIED_DIR.glob("*_classified.json"))
    total_sentences = 0
    classifications = []
    for f in classified_files:
        try:
            rows = json.loads(f.read_text())
            total_sentences += len(rows)
            for row in rows:
                parts = f.stem.replace("_classified", "").split("_")
                classifications.append({
                    "cik": parts[0],
                    "year_new": int(parts[2]),
                    **row
                })
        except Exception:
            pass

    data["n_classified_diffs"] = len(classified_files)
    data["total_sentences"] = total_sentences
    data["classifications"] = pd.DataFrame(classifications) if classifications else pd.DataFrame()

    return data


data = load_data()


def make_layout():
    n_diffs = data.get("n_classified_diffs", 0)
    n_sentences = data.get("total_sentences", 0)
    clf = data.get("classifications", pd.DataFrame())
    firms = data.get("firms", pd.DataFrame())

    # ── KPI cards ──
    kpis = dbc.Row([
        dbc.Col(dbc.Card([dbc.CardBody([html.H2(str(len(firms) if not firms.empty else 50), className="card-title text-primary"), html.P("Pilot Firms")])]), width=3),
        dbc.Col(dbc.Card([dbc.CardBody([html.H2(str(n_diffs), className="card-title text-success"), html.P("Diffs Classified")])]), width=3),
        dbc.Col(dbc.Card([dbc.CardBody([html.H2(f"{n_sentences:,}", className="card-title text-info"), html.P("Sentences Classified")])]), width=3),
        dbc.Col(dbc.Card([dbc.CardBody([html.H2("305", className="card-title text-warning"), html.P("Total Diffs")])]), width=3),
    ], className="mb-4")

    charts = []

    if not clf.empty:
        # Risk type distribution
        rt_counts = clf["risk_type"].value_counts().reset_index()
        rt_counts.columns = ["risk_type", "count"]
        fig_rt = px.bar(rt_counts, x="risk_type", y="count", color="risk_type",
                        title="Risk Type Distribution", template="plotly_white")
        charts.append(dbc.Col(dcc.Graph(figure=fig_rt), width=6))

        # Nature distribution
        nat_counts = clf["nature"].value_counts().reset_index()
        nat_counts.columns = ["nature", "count"]
        fig_nat = px.pie(nat_counts, names="nature", values="count",
                         title="Nature of Change", template="plotly_white")
        charts.append(dbc.Col(dcc.Graph(figure=fig_nat), width=6))

        # Style distribution
        sty_counts = clf["style"].value_counts().reset_index()
        sty_counts.columns = ["style", "count"]
        fig_sty = px.pie(sty_counts, names="style", values="count",
                         title="Language Style (Concrete vs Vague)", template="plotly_white",
                         color_discrete_map={"concrete": "#2ecc71", "vague": "#e74c3c"})
        charts.append(dbc.Col(dcc.Graph(figure=fig_sty), width=6))

        # Risk type by year
        if "year_new" in clf.columns:
            yr_rt = clf.groupby(["year_new", "risk_type"]).size().reset_index(name="count")
            fig_yr = px.bar(yr_rt, x="year_new", y="count", color="risk_type", barmode="stack",
                            title="Risk Type Mentions by Year", template="plotly_white")
            charts.append(dbc.Col(dcc.Graph(figure=fig_yr), width=6))

    # Firm-year variables table
    vars_table = []
    if not data.get("vars", pd.DataFrame()).empty:
        df = data["vars"][["ticker", "year_new", "risk_update_intensity", "vagueness_ratio",
                            "boilerplate_ratio", "similarity", "sector"]].head(20)
        vars_table = [
            html.H5("Firm-Year Variables (preview)", className="mt-4"),
            dash_table.DataTable(
                data=df.round(3).to_dict("records"),
                columns=[{"name": c, "id": c} for c in df.columns],
                style_table={"overflowX": "auto"},
                style_cell={"fontSize": 12, "padding": "5px"},
                style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
                page_size=10,
            )
        ]

    return dbc.Container([
        html.H2("Risk Disclosure Research — Pilot Dashboard", className="my-4"),
        html.Hr(),
        kpis,
        dbc.Row(charts),
        *vars_table,
    ], fluid=True)


app.layout = make_layout()

if __name__ == "__main__":
    app.run(debug=True, port=8050, host="0.0.0.0")
