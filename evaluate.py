"""Evaluation set.  Run:  python evaluate.py

For each question we check three things:
  plan     the question was turned into the expected query plan (or refused)
  math     the top result matches an independent pandas calculation
  grounded every number in the written answer exists in the query result
"""
import sys
from datetime import date, timedelta

import pandas as pd

import engine
import llm
from pipeline import answer

OG, AU, BK, TE, HC = "Oil & Gas", "Automobiles", "Banks", "Technology", "Healthcare"
CASES = [
    ("Top 10 performers in oil and gas over the last 3 months", "top_performers", [OG], 10, "3M"),
    ("Show me the top 5 tech stocks over the past year", "top_performers", [TE], 5, "1Y"),
    ("Best 3 banks year to date", "top_performers", [BK], 3, "YTD"),
    ("Who led healthcare over the last 6 months?", "top_performers", [HC], 10, "6M"),
    ("Top 10 automobile names last month", "top_performers", [AU], 10, "1M"),
    ("Top energy performers this quarter", "top_performers", [OG], 10, "3M"),
    ("Compare the oil industry's top 10 performers against automobiles over 3 months", "compare_industries", [OG, AU], 10, "3M"),
    ("Top 5 banks versus top 5 tech over 6 months", "compare_industries", [BK, TE], 5, "6M"),
    ("Compare healthcare and technology over the past year", "compare_industries", [HC, TE], 10, "1Y"),
    ("Give me the trending for the top 10 tickers in oil over 3 months", "trend", [OG], 10, "3M"),
    ("How are the top 5 bank stocks trending this year?", "trend", [BK], 5, "YTD"),
    ("How did EV makers do over the last 6 months?", "top_performers", [AU], 10, "6M"),
    ("Which chipmakers led over the past 12 months?", "top_performers", [TE], 10, "1Y"),
    ("Top 3 pharma names this quarter", "top_performers", [HC], 3, "3M"),
    ("Top crypto performers this month", "refuse", None, None, None),
    ("Top banks over the last 2 years", "refuse", None, None, None),
    ("Best tech stocks over the last 10 days", "refuse", None, None, None),
    ("Which tech stock should I buy?", "refuse", None, None, None),
    ("What will oil stocks do next year?", "refuse", None, None, None),
    ("Top 10 real estate performers over 3 months", "refuse", None, None, None),
    ("Top 500 tech stocks over 3 months", "refuse", None, None, None),
]


def independent_top(con, industry, window):
    """Recompute the best performer with pandas only, no shared code path with engine.run."""
    px = pd.read_sql("SELECT p.*, t.industry FROM prices p JOIN tickers t USING(ticker)", con, parse_dates=["date"])
    end = px.date.max()
    start = pd.Timestamp(date(end.year, 1, 1)) if window == "YTD" else end - timedelta(days=engine.WINDOWS[window])
    w = px[(px.industry == industry) & (px.date >= start)].sort_values("date")
    g = w.groupby("ticker").close.agg(["first", "last"])
    r = ((g["last"] / g["first"] - 1) * 100).round(2).sort_values(ascending=False)
    return r.index[0], float(r.iloc[0])


def run_cases(con, client=None):
    """Yields one result row per test case. Shared by the CLI below and the app's Evaluation tab."""
    for q, intent, inds, n, win in CASES:
        out = answer(q, con, log=False, client=client)
        plan_ok = math_ok = grounded = None
        if intent == "refuse":
            ok = out["status"].startswith("refused")
            detail = out["status"]
        else:
            p = out.get("plan", {})
            plan_ok = (out["status"] == "answered" and p["intent"] == intent and sorted(p["industries"]) == sorted(inds)
                       and p["top_n"] == n and p["window"] == win)
            math_ok = grounded = False
            if out["status"] == "answered":
                t = out["facts"]["table"]
                first = t[t.industry == inds[0]].iloc[0]
                tk, ret = independent_top(con, inds[0], p["window"])
                math_ok = first.ticker == tk and abs(first.return_pct - ret) < 0.01
                grounded = not llm.number_check(out["text"], out["facts"])
            ok = plan_ok and math_ok and grounded
            detail = f"plan={'ok' if plan_ok else 'FAIL'} math={'ok' if math_ok else 'FAIL'} grounded={'ok' if grounded else 'FAIL'}"
        yield {"result": "PASS" if ok else "FAIL", "question": q, "expected": intent, "status": out["status"],
               "plan": plan_ok, "math": math_ok, "grounded": grounded, "detail": detail,
               "latency_ms": out["latency_ms"]}


def main():
    con = engine.load_db()
    client = llm.make_client()
    passed = 0
    print(f"Language layer: {'Claude ' + llm.MODEL if client else 'rules + template'}\n")
    for r in run_cases(con, client):
        passed += r["result"] == "PASS"
        print(f"{r['result']}  {r['detail']:38}  {r['question']}")
    print(f"\n{passed}/{len(CASES)} passed")
    sys.exit(0 if passed == len(CASES) else 1)


if __name__ == "__main__":
    main()
