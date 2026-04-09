"""
Script 05: Fetch WRDS Data
Pulls Compustat annual fundamentals and CRSP annual returns for the 50 pilot firms.

Requires WRDS credentials (will prompt on first run; saved to ~/.pgpass after that).

Outputs:
  outputs/compustat_panel.csv  — annual financials with derived ratios
  outputs/crsp_returns.csv     — annual stock returns
"""

import pandas as pd
import wrds
from pathlib import Path

PILOT_FIRMS_CSV = Path("/Users/timtonnaer/risk_project/outputs/all_firms.csv")
OUTPUT_COMPUSTAT = Path("/Users/timtonnaer/risk_project/outputs/compustat_panel.csv")
OUTPUT_CRSP = Path("/Users/timtonnaer/risk_project/outputs/crsp_returns.csv")

YEAR_MIN = 2014   # one year before pilot start to allow t-1 controls
YEAR_MAX = 2024   # one year after pilot end to capture t+1 outcomes


def fetch_compustat(db: wrds.Connection, cik_list: list[str]) -> pd.DataFrame:
    # Compustat stores CIKs zero-padded to 10 digits — include both formats
    padded = [c.zfill(10) for c in cik_list]
    all_ciks = list(set(cik_list) | set(padded))
    cik_sql = ", ".join(f"'{c}'" for c in all_ciks)
    query = f"""
        SELECT gvkey, cik, fyear, datadate,
               ni, at, sale, dltt, act, lct, ceq,
               sich
        FROM comp.funda
        WHERE cik IN ({cik_sql})
          AND fyear BETWEEN {YEAR_MIN} AND {YEAR_MAX}
          AND indfmt = 'INDL'
          AND datafmt = 'STD'
          AND popsrc = 'D'
          AND consol = 'C'
        ORDER BY cik, fyear
    """
    df = db.raw_sql(query, date_cols=["datadate"])

    # Derived variables
    import math, numpy as np
    df = df.copy()
    at = df["at"].replace(0, np.nan)
    df.loc[:, "roa"]           = df["ni"] / at
    df.loc[:, "leverage"]      = df["dltt"] / at
    df.loc[:, "current_ratio"] = df["act"] / df["lct"].replace(0, np.nan)
    df.loc[:, "log_assets"]    = at.apply(lambda x: math.log(x) if x > 0 else np.nan)

    # Revenue growth (within firm, sorted by year)
    df = df.sort_values(["cik", "fyear"])
    df.loc[:, "rev_growth"] = df.groupby("cik")["sale"].pct_change()

    return df


def fetch_crsp(db: wrds.Connection, ticker_list: list[str]) -> pd.DataFrame:
    ticker_sql = ", ".join(f"'{t}'" for t in ticker_list)
    # Compound daily returns to annual; join via dsenames to get tickers
    query = f"""
        SELECT b.ticker,
               b.permno,
               EXTRACT(YEAR FROM a.date)::int AS year,
               EXP(SUM(LN(1 + a.ret))) - 1 AS annual_ret,
               COUNT(*) AS trading_days
        FROM crsp.dsf a
        JOIN crsp.dsenames b
          ON a.permno = b.permno
         AND a.date BETWEEN b.namedt AND COALESCE(b.nameendt, CURRENT_DATE)
        WHERE b.ticker IN ({ticker_sql})
          AND a.date BETWEEN '{YEAR_MIN}-01-01' AND '{YEAR_MAX}-12-31'
          AND a.ret IS NOT NULL
        GROUP BY b.ticker, b.permno, EXTRACT(YEAR FROM a.date)
        ORDER BY b.ticker, year
    """
    return db.raw_sql(query)


def main():
    firms = pd.read_csv(PILOT_FIRMS_CSV)
    cik_list = firms["cik"].astype(str).tolist()
    ticker_list = firms["ticker"].tolist()

    print("Connecting to WRDS...")
    db = wrds.Connection()

    print(f"Fetching Compustat for {len(cik_list)} firms, {YEAR_MIN}-{YEAR_MAX}...")
    compustat = fetch_compustat(db, cik_list)
    compustat.to_csv(OUTPUT_COMPUSTAT, index=False)
    print(f"  → {len(compustat)} firm-year rows saved to {OUTPUT_COMPUSTAT}")

    print(f"Fetching CRSP returns for {len(ticker_list)} tickers, {YEAR_MIN}-{YEAR_MAX}...")
    crsp = fetch_crsp(db, ticker_list)
    crsp.to_csv(OUTPUT_CRSP, index=False)
    print(f"  → {len(crsp)} ticker-year rows saved to {OUTPUT_CRSP}")

    db.close()

    print("\nCompustat summary:")
    print(compustat[["roa", "rev_growth", "leverage", "current_ratio"]].describe().round(3))

    print("\nCRSP summary:")
    print(crsp[["annual_ret", "trading_days"]].describe().round(3))


if __name__ == "__main__":
    main()
