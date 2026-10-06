"""Portfolio Q&A.  Run:  streamlit run app.py

Config (environment variables or .streamlit/secrets.toml):
  ANTHROPIC_API_KEY  optional; without it the rule parser and template writer are used
  APP_PASSWORD       optional; when set, visitors must enter it before using the app
"""
import hmac
import html
import json
import os
import time
import uuid

import altair as alt
import pandas as pd
import streamlit as st

import engine
import evaluate
import llm
from pipeline import answer, read_audit

st.set_page_config(page_title="Portfolio Q&A", page_icon=":material/query_stats:", layout="wide",
                   initial_sidebar_state="collapsed")

MAX_QUESTION_CHARS = 300
MAX_MODEL_QUESTIONS = 40  # per session, only when a model key is configured
COLORS = {"Automobiles": "#0284c7", "Banks": "#15803d", "Healthcare": "#be123c",
          "Oil & Gas": "#b45309", "Technology": "#6d28d9"}
SHORT = {"Oil & Gas": "oil and gas", "Automobiles": "auto", "Banks": "bank",
         "Technology": "tech", "Healthcare": "healthcare"}
PLURAL = {"Oil & Gas": "oil and gas", "Automobiles": "autos", "Banks": "banks",
          "Technology": "tech", "Healthcare": "healthcare"}
PHRASE = {"1M": "over the last month", "3M": "over the last 3 months", "6M": "over the last 6 months",
          "1Y": "over the past year", "YTD": "this year"}
EXAMPLES = [
    "Top 10 oil and gas names over the last 3 months",
    "Top 5 banks versus top 5 tech over 6 months",
    "How are the top 5 bank stocks trending this year?",
    "Who led healthcare over the past year?",
    "Top crypto performers this month",
    "Which tech stock should I buy?",
]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="st-"], .stMarkdown, button, input, textarea {font-family: 'IBM Plex Sans', sans-serif;}
code, pre, [data-testid="stMetricValue"] {font-family: 'IBM Plex Mono', monospace !important;}
[data-testid="stIconMaterial"], .material-symbols-rounded {font-family: 'Material Symbols Rounded' !important;}
[data-testid="stBaseButton-secondary"] p {white-space: normal; text-align: left; overflow: visible; text-overflow: clip;}
[data-testid="stToolbar"], [data-testid="stAppDeployButton"], [data-testid="stMainMenu"], footer,
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"] {display: none !important;}
header[data-testid="stHeader"] {background: transparent; height: 0;}
.block-container {padding-top: 2.4rem; max-width: 1120px;}
.top {display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem;}
.eyebrow {font-size: .75rem; letter-spacing: .08em; text-transform: uppercase; color: #64748b; font-weight: 500;}
.top h1 {font-size: 2.1rem; font-weight: 600; margin: .15rem 0 .4rem; padding: 0; letter-spacing: -.01em;}
.lede {max-width: 46rem; color: #334155; line-height: 1.55; margin: 0;}
.byline {font-size: .85rem; color: #64748b; white-space: nowrap; padding-top: .2rem;}
.coverage {font-size: .82rem; color: #64748b; margin: .7rem 0 .2rem;}
.steps {display: flex; flex-wrap: wrap; gap: .35rem; align-items: center; margin: .9rem 0 1.4rem;}
.step {border: 1px solid #cbd5e1; border-radius: 4px; padding: .18rem .55rem; font-size: .78rem; color: #334155; background: #f8fafc;}
.step.llm {border-color: #c4b5fd; background: #f5f3ff;}
.step i {font-style: normal; color: #94a3b8; margin-right: .3rem;}
.sep {color: #cbd5e1; font-size: .75rem;}
.q {font-weight: 600; font-size: 1.05rem; margin-bottom: .1rem;}
.readas {font-size: .78rem; color: #64748b; margin-bottom: .6rem;}
.badges {display: flex; flex-wrap: wrap; gap: .3rem; margin: .5rem 0 .8rem;}
.badge {font-size: .72rem; padding: .1rem .5rem; border-radius: 3px; border: 1px solid #e2e8f0; color: #475569;}
.badge.ok {background: #f0fdf4; border-color: #bbf7d0; color: #166534;}
.badge.warn {background: #fffbeb; border-color: #fde68a; color: #92400e;}
.refusal {border-left: 3px solid #f59e0b; background: #fffbeb; padding: .7rem .9rem; border-radius: 0 4px 4px 0; color: #78350f;}
.foot {font-size: .75rem; color: #94a3b8; margin-top: 2.5rem; border-top: 1px solid #e2e8f0; padding-top: .8rem;}
[data-testid="stMetricValue"] > div {font-size: clamp(1.2rem, 2.1vw, 1.8rem);}
[data-testid="stMetricLabel"] p {white-space: normal;}
@media (max-width: 640px) {
  .top {flex-direction: column-reverse; gap: .2rem;}
  .top h1 {font-size: 1.7rem;}
  .block-container {padding-top: 1.2rem;}
}
</style>
""", unsafe_allow_html=True)


def setting(name):
    """Environment first (Railway, Docker), then .streamlit/secrets.toml. Never sent to the browser."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        return st.secrets.get(name)
    except Exception:  # no secrets file
        return None


def db():
    """One in-memory database per visitor session: a sqlite connection must not be shared across threads."""
    if "con" not in st.session_state:
        st.session_state.con = engine.load_db()
    return st.session_state.con


@st.cache_resource
def _client_for(key):
    return llm.make_client(key)


def model_client():
    # keyed on the current secret, so a key added or rotated after start-up is picked up
    return _client_for(setting("ANTHROPIC_API_KEY"))


# ---------------------------------------------------------------- password gate

def hosted():
    """Streamlit Community Cloud checks the repo out under /mount/src."""
    return os.getcwd().startswith("/mount/src") or bool(os.environ.get("REQUIRE_PASSWORD"))


def require_password():
    expected = setting("APP_PASSWORD")
    if not expected and hosted():  # never serve a hosted copy unprotected
        st.markdown("<div style='height:15vh'></div><h3 style='text-align:center;font-weight:600'>Portfolio Q&A</h3>"
                    "<p style='text-align:center;color:#64748b'>This demo isn't open yet.</p>", unsafe_allow_html=True)
        st.stop()
    if not expected or st.session_state.get("authed"):
        return
    st.session_state.setdefault("fails", 0)
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.markdown("<div style='height:12vh'></div><div class='eyebrow'>Private demo</div>"
                    "<h2 style='margin:.2rem 0 .3rem;font-weight:600'>Portfolio Q&A</h2>"
                    "<p style='color:#64748b;margin-bottom:1rem'>Enter the password from my email to continue.</p>",
                    unsafe_allow_html=True)
        with st.form("login"):
            pw = st.text_input("Password", type="password", label_visibility="collapsed", placeholder="Password")
            ok = st.form_submit_button("Continue", type="primary", width="stretch")
        if ok:
            if st.session_state.fails >= 5:
                time.sleep(3)  # slow down guessing
            if hmac.compare_digest(pw.encode(), expected.encode()):
                st.session_state.authed = True
                st.rerun()
            st.session_state.fails += 1
            st.error("That password doesn't match.")
    st.stop()


require_password()

con = db()
client = model_client()
inds = engine.industries(con)
n_tickers = con.execute("SELECT COUNT(*) FROM tickers").fetchone()[0]
st.session_state.setdefault("sid", uuid.uuid4().hex[:12])
st.session_state.setdefault("history", [])
st.session_state.setdefault("asked", 0)

# ---------------------------------------------------------------- header

layer = f"Claude ({llm.MODEL})" if client else "rules and templates (no model key configured)"
steps = [("llm", "Model reads the question"), ("", "Scope check"), ("", "Fixed SQL"),
         ("", "Data checks"), ("llm", "Model writes it up"), ("", "Number check")]
st.markdown(f"""
<div class="top">
  <div>
    <div class="eyebrow">Governed Q&A over market data</div>
    <h1>Portfolio Q&A</h1>
    <p class="lede">Ask about industry performance in plain English. The numbers come from SQL, never from the
    model. The model only turns your question into a query plan and writes up results that have already been
    checked.</p>
  </div>
  <div class="byline">Ashish Shiwlani</div>
</div>
<div class="coverage">{n_tickers} tickers &middot; {len(inds)} industries &middot; daily closes through
{engine.as_of(con):%b %-d, %Y} &middot; language layer: {layer}</div>
<div class="steps">{'<span class="sep">&rarr;</span>'.join(
    f'<span class="step {k}"><i>{i}</i>{t}</span>' for i, (k, t) in enumerate(steps, 1))}
<span style="font-size:.72rem;color:#94a3b8;margin-left:.4rem">purple = model, grey = deterministic</span></div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- answer rendering

def pct(v):
    return f"{v:+.2f}%"


def bar_chart(table):
    d = table.assign(label=table.return_pct.map(pct))
    order = list(d.sort_values("return_pct", ascending=False).ticker)
    present = [i for i in COLORS if i in set(d.industry)]
    base = alt.Chart(d).encode(
        y=alt.Y("ticker:N", sort=order, title=None, axis=alt.Axis(labelFontSize=12, labelOverlap=False)),
        x=alt.X("return_pct:Q", title="Return (%)"),
        tooltip=["industry", "ticker", alt.Tooltip("start_close:Q", format="$.2f"),
                 alt.Tooltip("end_close:Q", format="$.2f"), alt.Tooltip("return_pct:Q", format="+.2f")])
    bars = base.mark_bar(cornerRadiusEnd=2, height={"band": 0.72}).encode(
        color=alt.Color("industry:N", scale=alt.Scale(domain=present, range=[COLORS[i] for i in present]),
                        legend=alt.Legend(orient="bottom", title=None) if len(present) > 1 else None))
    text = base.mark_text(align="left", dx=4, fontSize=11, color="#475569").encode(text="label:N")
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#94a3b8").encode(x="x:Q")
    return (bars + text + zero).properties(height=max(200, 28 * len(d)))


def trend_chart(series):
    d = series.reset_index().melt(id_vars="date", var_name="ticker", value_name="index")
    d["date"] = pd.to_datetime(d["date"])
    lines = alt.Chart(d).mark_line(strokeWidth=1.6).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("index:Q", title="Price, start of window = 100", scale=alt.Scale(zero=False)),
        color=alt.Color("ticker:N", legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["ticker", alt.Tooltip("date:T"), alt.Tooltip("index:Q", format=".1f")])
    base = alt.Chart(pd.DataFrame({"y": [100]})).mark_rule(strokeDash=[4, 4], color="#94a3b8").encode(y="y:Q")
    return (lines + base).properties(height=340)


TABLE_CONFIG = {
    "industry": "Industry", "rank": st.column_config.NumberColumn("Rank", width="small"),
    "ticker": st.column_config.TextColumn("Ticker", width="small"),
    "start_close": st.column_config.NumberColumn("Start", format="$%.2f"),
    "end_close": st.column_config.NumberColumn("Latest", format="$%.2f"),
    "return_pct": st.column_config.NumberColumn("Return", format="%+.2f%%"),
}
SUMMARY_CONFIG = {
    "industry": "Industry", "tickers": "Names",
    "avg_return_pct": st.column_config.NumberColumn("Average", format="%+.2f%%"),
    "median_return_pct": st.column_config.NumberColumn("Median", format="%+.2f%%"),
    "best_ticker": "Leader",
    "best_return_pct": st.column_config.NumberColumn("Best", format="%+.2f%%"),
    "worst_return_pct": st.column_config.NumberColumn("Worst", format="%+.2f%%"),
}


def read_as(p):
    kind = {"top_performers": f"Top {p['top_n']}", "compare_industries": f"Top {p['top_n']} in each",
            "trend": f"Price trend, top {p['top_n']}"}[p["intent"]]
    return f"Read as: {kind} &middot; {html.escape(' vs '.join(p['industries']))} &middot; {engine.WINDOW_LABEL[p['window']]}"


def followups(p):
    n, w = p["top_n"], p["window"]
    names = " versus ".join(SHORT.get(i, i.lower()) for i in p["industries"])
    other_w = "1Y" if w != "1Y" else "3M"
    if p["intent"] == "compare_industries" or len(p["industries"]) > 1:
        plural = " versus ".join(PLURAL.get(i, i.lower()) for i in p["industries"])
        return [f"Compare top {n} {plural} {PHRASE[other_w]}"]
    if p["intent"] == "trend":
        return [f"Top {n} {names} names {PHRASE[w]}", f"How are the top {n} {names} names trending {PHRASE[other_w]}?"]
    other = next(i for i in inds if i not in p["industries"])
    return [f"Top {n} {names} names {PHRASE[other_w]}",
            f"How are the top {min(n, 5)} {names} names trending {PHRASE[w]}?",
            f"Compare top {min(n, 5)} {PLURAL.get(p['industries'][0])} versus {PLURAL.get(other, other.lower())} {PHRASE[w]}"]


def ask_next(q):
    st.session_state.pending = q


def badges(out):
    f = out["facts"]
    items = [("ok", "Numbers match the query result") if not out["rejected_numbers"]
             else ("warn", f"Model draft rejected ({', '.join(out['rejected_numbers'])} not in the data), template used")]
    excluded = sorted({t for c in f["checks"] if c["status"] == "excluded" for t in c["tickers"]})
    items.append(("warn", f"Left out by data checks: {', '.join(excluded)}") if excluded else ("ok", "Data checks passed"))
    items += [("", f"Parsed by {out['parser']}"), ("", f"Written by {out['writer']}"), ("", f"{out['latency_ms']:,} ms")]
    st.markdown("<div class='badges'>" + "".join(f"<span class='badge {k}'>{html.escape(t)}</span>" for k, t in items)
                + "</div>", unsafe_allow_html=True)


def render_answer(q, out, key):
    st.markdown(f"<div class='q'>{html.escape(q)}</div>", unsafe_allow_html=True)
    if out["status"] != "answered":
        reason = ("Stopped at the scope check, so no query was run." if "hard" in out["status"]
                  else "There wasn't enough clean data to answer.")
        st.markdown(f"<div class='readas'>Parsed by {html.escape(out['parser'])}</div>"
                    f"<div class='refusal'>{html.escape(out['text'])}<br><span style='font-size:.8rem;opacity:.8'>"
                    f"{reason}</span></div>", unsafe_allow_html=True)
        with st.expander("Query plan"):
            st.json(out["plan"])
        return

    f = out["facts"]
    p, table, summary = f["plan"], f["table"], f["summary"]
    st.markdown(f"<div class='readas'>{read_as(p)}</div>", unsafe_allow_html=True)
    st.markdown(out["text"].replace("$", "\\$"))  # "$" would otherwise start a LaTeX block
    badges(out)

    cols = st.columns(len(summary) + 1) if len(summary) <= 3 else st.columns(len(summary))
    for c, s in zip(cols, summary.itertuples()):
        c.metric(f"{s.industry}, average of top {s.tickers}", pct(s.avg_return_pct),
                 f"{s.best_ticker} {pct(s.best_return_pct)}", delta_color="off", delta_arrow="off", border=True)
    if len(summary) <= 3:
        cols[-1].metric("Window", engine.WINDOW_LABEL[p["window"]],
                        f"{pd.Timestamp(f['window_start']):%b %-d} to {pd.Timestamp(f['as_of']):%b %-d}",
                        delta_color="off", delta_arrow="off", border=True)

    left, right = st.columns(2)
    with left:
        st.altair_chart(trend_chart(f["series"]) if p["intent"] == "trend" else bar_chart(table), width="stretch")
    with right:
        st.dataframe(table, hide_index=True, width="stretch", column_config=TABLE_CONFIG,
                     height=min(36 * (len(table) + 1) + 3, 420))
        st.download_button("Download CSV", table.to_csv(index=False), "portfolio_answer.csv", "text/csv",
                           key=f"dl{key}", icon=":material/download:", type="tertiary")
    if len(summary) > 1:
        st.dataframe(summary, hide_index=True, width="stretch", column_config=SUMMARY_CONFIG)

    with st.expander("How this answer was produced"):
        st.markdown(f"**1. Question parsed by** `{out['parser']}` into this plan, then validated:")
        st.json(p)
        st.markdown("**2. Fixed SQL template with bound parameters.** The model never writes SQL.")
        st.code(f["sql"], language="sql")
        st.code(json.dumps(f["sql_params"]), language="json")
        st.markdown("**3. Data quality checks**")
        st.dataframe(pd.DataFrame(f["checks"]).assign(tickers=lambda d: d.tickers.map(", ".join)),
                     hide_index=True, width="stretch")
        st.markdown(f"**4. Written by** `{out['writer']}`")
        if out["rejected_numbers"]:
            st.warning(f"The model's draft contained numbers that are not in the result {out['rejected_numbers']}, "
                       "so it was thrown away and the deterministic template was used.")
        else:
            st.success("Every number in the answer appears in the query result.")
        st.markdown(f"**5. Metric.** {f['metric_definition']}")

    st.caption("Ask next")
    nxt = st.columns(3)
    for i, fq in enumerate(followups(p)):
        nxt[i].button(fq, key=f"fu{key}_{i}", on_click=ask_next, args=(fq,), width="stretch", type="secondary")


# ---------------------------------------------------------------- tabs

ask_tab, test_tab, log_tab, method_tab = st.tabs(["Ask", "Test set", "Audit log", "Method"])

with ask_tab:
    with st.form("ask", clear_on_submit=True, border=False):
        c1, c2 = st.columns([6, 1], vertical_alignment="bottom")
        typed = c1.text_input("Question", placeholder="e.g. top 5 banks versus tech over 6 months",
                              label_visibility="collapsed", max_chars=MAX_QUESTION_CHARS)
        sent = c2.form_submit_button("Ask", type="primary", width="stretch")

    def use_example():
        ask_next(st.session_state.example)
        st.session_state.example = None

    st.pills("Examples", EXAMPLES, key="example", on_change=use_example, label_visibility="collapsed")

    q = (typed.strip() if sent else "") or st.session_state.pop("pending", None)
    if q:
        if client and st.session_state.asked >= MAX_MODEL_QUESTIONS:
            st.info(f"This session has reached its limit of {MAX_MODEL_QUESTIONS} questions. Refresh to start a new one.")
        else:
            with st.spinner("Reading the question, querying, checking the numbers"):
                out = answer(q[:MAX_QUESTION_CHARS], con, client=client, session=st.session_state.sid)
            st.session_state.asked += 1
            st.session_state.history.insert(0, (q, out))

    if not st.session_state.history:
        st.markdown("<p style='color:#64748b;margin-top:1rem'>Pick an example or ask your own question. "
                    "It covers ranking, comparing two or more industries, and price trends. It will decline "
                    "forecasts, investment advice and anything outside the data.</p>", unsafe_allow_html=True)
    for i, (hq, hout) in enumerate(st.session_state.history):
        with st.container(border=True):
            render_answer(hq, hout, len(st.session_state.history) - i)

with test_tab:
    st.markdown(f"##### {len(evaluate.CASES)} questions with known answers")
    st.markdown("Each question is checked three ways. **Plan**: it was read into the expected query, or "
                "refused. **Math**: the top result matches a separate pandas calculation that shares no code "
                "with the engine. **Grounded**: every number in the written answer is in the query result.")
    if client:
        st.caption(f"This runs every question through {llm.MODEL}, so it takes a minute or two.")
    if st.button("Run the test set", type="primary"):
        rows, bar = [], st.progress(0.0, "Running")
        for i, r in enumerate(evaluate.run_cases(con, client), 1):
            rows.append(r)
            bar.progress(i / len(evaluate.CASES), r["question"])
        bar.empty()
        st.session_state.eval = (pd.DataFrame(rows), layer)
    if "eval" in st.session_state:
        df, mode = st.session_state.eval
        ans, ref = df[df.expected != "refuse"], df[df.expected == "refuse"]
        c = st.columns(4)
        c[0].metric("Passed", f"{(df.result == 'PASS').sum()} of {len(df)}", border=True)
        c[1].metric("Answered correctly", f"{(ans.result == 'PASS').sum()} of {len(ans)}", border=True)
        c[2].metric("Refused correctly", f"{(ref.result == 'PASS').sum()} of {len(ref)}", border=True)
        c[3].metric("Median time", f"{df.latency_ms.median():,.0f} ms", border=True)
        st.caption(f"Language layer: {mode}")
        mark = {True: "✓", False: "✗", None: ""}
        st.dataframe(df.assign(plan=df.plan.map(mark), math=df.math.map(mark), grounded=df.grounded.map(mark),
                               result=df.result.str.title())
                     [["result", "question", "expected", "plan", "math", "grounded", "latency_ms"]],
                     hide_index=True, width="stretch",
                     column_config={"result": "Result", "question": "Question", "expected": "Expected",
                                    "plan": "Plan", "math": "Math", "grounded": "Grounded",
                                    "latency_ms": st.column_config.NumberColumn("Time", format="%d ms")})

with log_tab:
    log = read_audit(st.session_state.sid)
    st.markdown("##### Your requests in this session")
    st.markdown("Each request is written to an append-only log with the question, the parsed plan, the SQL "
                "parameters, data checks, who wrote the answer, any rejected numbers, a hash of the result set "
                "and the time taken. The same question over the same data always produces the same hash.")
    if not log:
        st.caption("Nothing yet. Ask something first.")
    else:
        df = pd.DataFrame(log)
        for col in ("writer", "result_hash", "latency_ms"):
            if col not in df:
                df[col] = None
        answered = df.status == "answered"
        fell_back = df[answered].writer.fillna("").str.contains("failed|no draft")
        c = st.columns(4)
        c[0].metric("Requests", len(df), border=True)
        c[1].metric("Refused", f"{(~answered).mean():.0%}", border=True)
        c[2].metric("Model draft replaced", f"{fell_back.mean() if len(fell_back) else 0:.0%}", border=True,
                    help="Answered questions where the model's draft failed the number check or the call failed.")
        c[3].metric("Median time", f"{df.latency_ms.median():,.0f} ms", border=True)
        st.dataframe(df[["ts", "question", "status", "parser", "writer", "latency_ms", "result_hash"]],
                     hide_index=True, width="stretch",
                     column_config={"ts": "Time (UTC)", "question": "Question", "status": "Status",
                                    "parser": "Parser", "writer": "Writer", "result_hash": "Result hash",
                                    "latency_ms": st.column_config.NumberColumn("Time", format="%d ms")})
        st.download_button("Download as JSON lines", "\n".join(json.dumps(r) for r in reversed(log)),
                           "audit_log.jsonl", icon=":material/download:", type="tertiary")

with method_tab:
    st.graphviz_chart("""
    digraph {
      rankdir=TB; bgcolor="transparent"; nodesep=0.35; ranksep=0.45;
      node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=13 color="#cbd5e1"
            fillcolor="#f8fafc" fontcolor="#0f172a" margin="0.18,0.08"];
      edge [color="#94a3b8" fontname="Helvetica" fontsize=11 fontcolor="#64748b"];
      q [label="Question"];
      p [label="1  Model reads the question\\ninto a fixed-schema plan" fillcolor="#f5f3ff" color="#c4b5fd"];
      g [label="2  Scope check\\nindustries, limits, windows"];
      s [label="3  Fixed SQL template\\nbound parameters"];
      d [label="4  Data checks\\nmissing, stale, invalid"];
      w [label="5  Model writes it up\\nfrom checked facts only" fillcolor="#f5f3ff" color="#c4b5fd"];
      n [label="6  Number check\\nmismatch: use template"];
      a [label="Answer, table, chart\\nand audit record" fillcolor="#f0fdf4" color="#bbf7d0"];
      r [label="Refusal\\nno query run" fillcolor="#fffbeb" color="#fde68a"];
      {rank=same; q; p; g; s}
      {rank=same; r; d; w; n; a}
      q -> p -> g -> s; s -> d; d -> w -> n -> a;
      g -> r [label=" out of scope" style=dashed];
    }""", width="stretch")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("""
##### Why it's built this way

In finance a fluent answer with one wrong number is worse than no answer, so the model is kept away from
the arithmetic. It picks from a closed menu of industries, windows and limits. A fixed SQL template does
the calculation, and the result is checked for missing, stale and invalid prices before anyone sees it.

The model then writes a short summary from those checked facts. Every number in that summary is matched
back to the result set, and if one doesn't match, the summary is replaced with a plain template. Questions
the data can't answer, such as forecasts, advice or industries that aren't loaded, are refused before a
query runs.

Without a model key it still works. A keyword parser and the template writer take over, which is also
what happens if the API times out.
""")
    with c2:
        st.markdown("""
##### What I'd change for production

- Route model calls through the firm's LLM gateway rather than a public API
- Enforce each user's data entitlements in the SQL layer
- Move metric definitions into a governed semantic layer that reporting also uses
- Send failed-check or low-confidence answers to a reviewer
- Track refusal rate, fallback rate, latency and cost per answered question
""")
    st.caption("Data: one year of public daily closes from Yahoo Finance for 60 tickers in 5 industries. "
               "BK is in the reference list with no prices on purpose, so the missing-data check has something real to catch.")

st.markdown("<div class='foot'>Ashish Shiwlani &middot; Portfolio Q&A &middot; sample data only, not investment advice</div>",
            unsafe_allow_html=True)
