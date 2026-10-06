"""Deterministic layer: the only place numbers are produced.

The LLM never touches this math. It hands us a small query plan, we validate it,
run a fixed SQL template with bound parameters, check the data, and return facts.
"""
import sqlite3
from datetime import date, timedelta

import pandas as pd

WINDOWS = {"1M": 30, "3M": 91, "6M": 182, "1Y": 365, "YTD": None}
WINDOW_LABEL = {"1M": "1 month", "3M": "3 months", "6M": "6 months", "1Y": "1 year", "YTD": "year to date"}
INTENTS = ["top_performers", "compare_industries", "trend", "unsupported"]
MAX_TOP_N = 25

RETURNS_SQL = """
WITH w AS (
  SELECT p.ticker, t.industry, p.date, p.close
  FROM prices p JOIN tickers t ON t.ticker = p.ticker
  WHERE t.industry IN ({ph}) AND p.date >= ?
),
f AS (
  SELECT ticker, industry, MIN(date) AS first_date, MAX(date) AS last_date
  FROM w GROUP BY ticker, industry
)
SELECT f.ticker, f.industry,
       f.first_date, a.close AS start_close,
       f.last_date,  b.close AS end_close,
       ROUND((b.close / a.close - 1) * 100, 2) AS return_pct
FROM f
JOIN prices a ON a.ticker = f.ticker AND a.date = f.first_date
JOIN prices b ON b.ticker = f.ticker AND b.date = f.last_date
"""


class Refusal(Exception):
    """Raised when the question cannot be answered from governed data."""

    def __init__(self, kind, message):
        super().__init__(message)
        self.kind, self.message = kind, message


def load_db(data_dir="data"):
    con = sqlite3.connect(":memory:", check_same_thread=False)
    pd.read_csv(f"{data_dir}/tickers.csv").to_sql("tickers", con, index=False)
    pd.read_csv(f"{data_dir}/prices.csv").to_sql("prices", con, index=False)
    con.execute("CREATE INDEX ix ON prices(ticker, date)")
    return con


def industries(con):
    return [r[0] for r in con.execute("SELECT DISTINCT industry FROM tickers ORDER BY 1")]


def as_of(con):
    return date.fromisoformat(con.execute("SELECT MAX(date) FROM prices").fetchone()[0])


def validate_plan(plan, con):
    """Pre-flight guardrail. Reject anything outside what the data can answer."""
    if plan.get("intent") not in INTENTS or plan["intent"] == "unsupported":
        raise Refusal("hard", plan.get("reason") or "This question is outside what this dataset covers.")
    known = industries(con)
    wanted = plan.get("industries") or []
    bad = [i for i in wanted if i not in known]
    if bad or not wanted:
        raise Refusal("hard", f"I only have data for these industries: {', '.join(known)}.")
    if plan.get("window") not in WINDOWS:
        raise Refusal("hard", f"Supported time windows are: {', '.join(WINDOW_LABEL.values())}.")
    n = plan.get("top_n", 10)
    if not isinstance(n, int) or not 1 <= n <= MAX_TOP_N:
        raise Refusal("hard", f"Top N must be between 1 and {MAX_TOP_N}.")
    intent = plan["intent"]
    if intent == "compare_industries" and len(wanted) < 2:
        intent = "top_performers"  # "compare the top oil names" is a ranking within one industry
    return {"intent": intent, "industries": wanted, "window": plan["window"], "top_n": n}


def run(plan, con):
    """Validate, query, check data quality, rank. Returns a dict of facts."""
    plan = validate_plan(plan, con)
    end = as_of(con)
    days = WINDOWS[plan["window"]]
    start = date(end.year, 1, 1) if days is None else end - timedelta(days=days)

    sql = RETURNS_SQL.format(ph=",".join("?" * len(plan["industries"])))
    params = [*plan["industries"], start.isoformat()]
    df = pd.read_sql(sql, con, params=params)

    checks = []
    # 1. reference tickers with no prices at all in the window
    ref = pd.read_sql(
        f"SELECT ticker FROM tickers WHERE industry IN ({','.join('?' * len(plan['industries']))})",
        con, params=plan["industries"])
    missing = sorted(set(ref.ticker) - set(df.ticker))
    if missing:
        checks.append({"check": "missing_prices", "status": "excluded", "tickers": missing,
                       "detail": "No price history in the data."})
    # 2. incomplete window: first price is more than a week after the window start
    late = df[pd.to_datetime(df.first_date).dt.date > start + timedelta(days=7)]
    if len(late):
        checks.append({"check": "incomplete_window", "status": "excluded", "tickers": sorted(late.ticker),
                       "detail": "Price history does not cover the full window."})
        df = df.drop(late.index)
    # 3. stale: last price is more than five days before the as-of date
    stale = df[pd.to_datetime(df.last_date).dt.date < end - timedelta(days=5)]
    if len(stale):
        checks.append({"check": "stale_price", "status": "excluded", "tickers": sorted(stale.ticker),
                       "detail": "Latest price is out of date."})
        df = df.drop(stale.index)
    # 4. bad values
    bad = df[(df.start_close <= 0) | (df.end_close <= 0) | df.return_pct.isna()]
    if len(bad):
        checks.append({"check": "invalid_price", "status": "excluded", "tickers": sorted(bad.ticker),
                       "detail": "Non-positive or missing price."})
        df = df.drop(bad.index)
    if not checks:
        checks.append({"check": "all", "status": "passed", "tickers": [], "detail": "No data quality issues."})

    if df.empty:
        raise Refusal("soft", "No tickers passed the data quality checks for that request.")

    df = df.sort_values(["industry", "return_pct"], ascending=[True, False])
    top = df.groupby("industry", sort=False).head(plan["top_n"]).reset_index(drop=True)
    top["rank"] = top.groupby("industry").cumcount() + 1
    for ind, g in top.groupby("industry"):
        if len(g) < plan["top_n"]:
            checks.append({"check": "fewer_than_requested", "status": "warning", "tickers": [],
                           "detail": f"{ind}: only {len(g)} tickers available, {plan['top_n']} requested."})

    summary = (top.groupby("industry", sort=False)
               .agg(tickers=("ticker", "count"),
                    avg_return_pct=("return_pct", "mean"),
                    median_return_pct=("return_pct", "median"),
                    best_ticker=("ticker", "first"),
                    best_return_pct=("return_pct", "max"),
                    worst_return_pct=("return_pct", "min"))
               .round(2).reset_index())

    series = pd.read_sql(
        f"SELECT date, ticker, close FROM prices WHERE ticker IN ({','.join('?' * len(top))}) AND date >= ?",
        con, params=[*top.ticker, start.isoformat()])
    series = series.pivot(index="date", columns="ticker", values="close")
    series = (series / series.bfill().iloc[0] * 100).round(2)

    return {
        "plan": plan,
        "metric_definition": "Performance = percent change in adjusted closing price from the first "
                             "trading day in the window to the latest trading day.",
        "window_start": start.isoformat(),
        "as_of": end.isoformat(),
        "sql": sql.strip(),
        "sql_params": params,
        "checks": checks,
        "table": top[["industry", "rank", "ticker", "start_close", "end_close", "return_pct"]],
        "summary": summary,
        "series": series,
    }
