"""End-to-end tests.  Run:  pip install pytest && pytest -q

engine   every intent x window x industry x top N, checked against an independent pandas calculation
parser   phrasings that should answer, and inputs that should be refused or must not crash
model    the Claude path against recorded-shape API responses, including failures (no network, no key)
app      the Streamlit UI driven headlessly: password gate, asking, examples, follow-ups, tabs, limits
"""
import itertools
import json
import os
import threading
from datetime import date, timedelta
from types import SimpleNamespace as NS

import anthropic
import httpx2 as httpx
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import engine
import llm
from pipeline import answer

CON = engine.load_db()
INDS = engine.industries(CON)
PX = pd.read_sql("SELECT p.*, t.industry FROM prices p JOIN tickers t USING(ticker)", CON, parse_dates=["date"])


@pytest.fixture(autouse=True)
def no_audit_file(tmp_path, monkeypatch):
    monkeypatch.setattr("pipeline.AUDIT_FILE", str(tmp_path / "audit.jsonl"))


def independent_returns(industry, window):
    end = PX.date.max()
    start = pd.Timestamp(date(end.year, 1, 1)) if window == "YTD" else end - timedelta(days=engine.WINDOWS[window])
    w = PX[(PX.industry == industry) & (PX.date >= start)].sort_values("date")
    g = w.groupby("ticker").close.agg(["first", "last"])
    return ((g["last"] / g["first"] - 1) * 100).round(2)


# ---------------------------------------------------------------- engine

@pytest.mark.parametrize("window", list(engine.WINDOWS))
@pytest.mark.parametrize("intent", ["top_performers", "trend"])
@pytest.mark.parametrize("industry", INDS)
@pytest.mark.parametrize("top_n", [1, 3, 10, 25])
def test_engine_matches_independent_math(industry, intent, window, top_n):
    f = engine.run({"intent": intent, "industries": [industry], "window": window, "top_n": top_n}, CON)
    t = f["table"]
    expected = independent_returns(industry, window)
    for r in t.itertuples():
        assert abs(r.return_pct - expected[r.ticker]) < 0.011, r.ticker
    assert list(t["rank"]) == list(range(1, len(t) + 1))
    assert t.return_pct.is_monotonic_decreasing
    assert len(t) == min(top_n, expected.size)
    assert (f["series"].bfill().iloc[0].round(1) == 100).all()
    assert llm.number_check(llm.write_template(f), f) == []


@pytest.mark.parametrize("pair", list(itertools.combinations(INDS, 2)) + [tuple(INDS)])
@pytest.mark.parametrize("window", list(engine.WINDOWS))
def test_engine_compare(pair, window):
    f = engine.run({"intent": "compare_industries", "industries": list(pair), "window": window, "top_n": 5}, CON)
    assert set(f["summary"].industry) == set(pair)
    assert llm.number_check(llm.write_template(f), f) == []


def test_missing_ticker_is_reported_not_ranked():
    f = engine.run({"intent": "top_performers", "industries": ["Banks"], "window": "1Y", "top_n": 25}, CON)
    assert "BK" not in set(f["table"].ticker)
    assert any(c["check"] == "missing_prices" and c["tickers"] == ["BK"] for c in f["checks"])


@pytest.mark.parametrize("plan", [
    {"intent": "unsupported", "industries": [], "top_n": 10, "window": "3M"},
    {"intent": "top_performers", "industries": ["Crypto"], "top_n": 10, "window": "3M"},
    {"intent": "top_performers", "industries": [], "top_n": 10, "window": "3M"},
    {"intent": "top_performers", "industries": ["Banks"], "top_n": 0, "window": "3M"},
    {"intent": "top_performers", "industries": ["Banks"], "top_n": 26, "window": "3M"},
    {"intent": "top_performers", "industries": ["Banks"], "top_n": "5", "window": "3M"},
    {"intent": "top_performers", "industries": ["Banks"], "top_n": 5, "window": "2Y"},
    {"intent": "drop_tables", "industries": ["Banks"], "top_n": 5, "window": "3M"},
    {},
])
def test_bad_plans_are_refused(plan):
    with pytest.raises(engine.Refusal):
        engine.run(plan, CON)


def test_single_industry_compare_becomes_ranking():
    f = engine.run({"intent": "compare_industries", "industries": ["Banks"], "window": "3M", "top_n": 5}, CON)
    assert f["plan"]["intent"] == "top_performers"


def test_injection_in_industry_name_never_reaches_sql():
    with pytest.raises(engine.Refusal):
        engine.run({"intent": "top_performers", "industries": ["Banks'); DROP TABLE prices;--"],
                    "window": "3M", "top_n": 5}, CON)
    assert CON.execute("SELECT COUNT(*) FROM prices").fetchone()[0] > 0


# ---------------------------------------------------------------- parser (no model)

ANSWERS = {
    "tell me about the past 6 months stocks": ("compare_industries", INDS, 5, "6M"),
    "How did the market do this year?": ("compare_industries", INDS, 5, "YTD"),
    "how did stocks do": ("compare_industries", INDS, 5, "3M"),
    "Top 10 performers in oil and gas over the last 3 months": ("top_performers", ["Oil & Gas"], 10, "3M"),
    "top 5 banks ytd": ("top_performers", ["Banks"], 5, "YTD"),
    "best 3 pharma names over the past 12 months": ("top_performers", ["Healthcare"], 3, "1Y"),
    "How did EV makers do over the last 6 months?": ("top_performers", ["Automobiles"], 10, "6M"),
    "chipmakers last month": ("top_performers", ["Technology"], 10, "1M"),
    "Top 5 banks versus top 5 tech over 6 months": ("compare_industries", ["Banks", "Technology"], 5, "6M"),
    "compare healthcare and energy this quarter": ("compare_industries", ["Healthcare", "Oil & Gas"], 10, "3M"),
    "compare all industries over the past year": ("compare_industries", INDS, 10, "1Y"),
    "How are the top 5 bank stocks trending this year?": ("trend", ["Banks"], 5, "YTD"),
    "show me the trend for refiners over 6 months": ("trend", ["Oil & Gas"], 10, "6M"),
    "TOP 3 TECH STOCKS PAST YEAR": ("top_performers", ["Technology"], 3, "1Y"),
    "top-5 auto names, last 3 months": ("top_performers", ["Automobiles"], 5, "3M"),
}
REFUSALS = [
    "Top crypto performers this month", "Which tech stock should I buy?", "What will oil stocks do next year?",
    "Top 10 real estate performers over 3 months", "Top 500 tech stocks over 3 months",
    "Top banks over the last 2 years", "Best tech stocks over the last 10 days", "forecast healthcare",
    "should I sell my bank shares", "best gold miners last quarter", "how are bonds doing",
    "recommend an EV stock", "top 0 banks",
]
WEIRD = ["", "   ", "?", "🚀🚀🚀", "<script>alert(1)</script>", "'; DROP TABLE prices; --",
         "a" * 5000, "top 99999999999999999999 banks", "top -5 banks", "\x00\x01 banks", "银行 过去六个月"]


@pytest.mark.parametrize("q,expected", ANSWERS.items())
def test_parser_answers(q, expected):
    out = answer(q, CON)
    assert out["status"] == "answered", out["text"]
    p = out["plan"]
    assert (p["intent"], sorted(p["industries"]), p["top_n"], p["window"]) == \
           (expected[0], sorted(expected[1]), expected[2], expected[3])
    assert out["rejected_numbers"] == [] and llm.number_check(out["text"], out["facts"]) == []


@pytest.mark.parametrize("q", REFUSALS)
def test_parser_refuses(q):
    assert answer(q, CON)["status"].startswith("refused")


@pytest.mark.parametrize("q", WEIRD)
def test_weird_input_never_crashes(q):
    out = answer(q, CON)
    assert out["status"] in ("answered", "refused_hard", "refused_soft")
    assert CON.execute("SELECT COUNT(*) FROM prices").fetchone()[0] > 0


def test_concurrent_sessions():
    errors = []

    def worker(q):
        try:
            con = engine.load_db()
            for _ in range(5):
                assert answer(q, con)["status"] == "answered"
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(q,)) for q in list(ANSWERS)[:8]]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors


# ---------------------------------------------------------------- model path, against the real SDK with a mock transport

def msg(content, stop="end_turn"):
    return {"id": "msg_test", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": content, "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 10}}


def tool(plan):
    return msg([{"type": "tool_use", "id": "toolu_1", "name": "make_query_plan", "input": plan}], "tool_use")


def text(t):
    return msg([{"type": "text", "text": t}])


class FakeAPI:
    """Replays responses in order and records every request body."""

    def __init__(self, *responses):
        self.responses, self.requests = list(responses), []

    def __call__(self, request):
        self.requests.append({"url": str(request.url), "headers": dict(request.headers),
                              "body": json.loads(request.content)})
        status, body = self.responses.pop(0)
        return httpx.Response(status, json=body)

    def client(self):
        return anthropic.Anthropic(api_key="sk-test", max_retries=0,
                                   http_client=httpx.Client(transport=httpx.MockTransport(self)))


PLAN = {"intent": "top_performers", "industries": ["Oil & Gas"], "top_n": 3, "window": "3M"}
FACTS = engine.run(PLAN, CON)
LEAD = FACTS["table"].iloc[0]


def test_model_happy_path_and_request_shape():
    prose = (f"{LEAD.ticker} led Oil & Gas over 3 months with {LEAD.return_pct}%, moving from "
             f"${LEAD.start_close} to ${LEAD.end_close} between {FACTS['window_start']} and {FACTS['as_of']}.")
    api = FakeAPI((200, tool(PLAN)), (200, text(prose)))
    out = answer("top 3 energy names this quarter", CON, client=api.client())
    assert out["status"] == "answered" and out["parser"] == out["writer"] == "claude-opus-5-5"
    assert out["text"] == prose and out["rejected_numbers"] == []
    parse, write = (r["body"] for r in api.requests)
    assert parse["model"] == llm.MODEL and parse["fallbacks"] == "default"
    assert parse["tools"][0]["strict"] is True and "tool_choice" not in parse
    assert "server-side-fallback-2026-07-01" in api.requests[0]["headers"]["anthropic-beta"]
    assert "FACTS" in write["messages"][0]["content"]


def test_written_out_dates():
    end = pd.Timestamp(FACTS["as_of"])
    good = f"{LEAD.ticker} led at {LEAD.return_pct}% as of {end:%B} {end.day}, {end.year}."
    bad = f"{LEAD.ticker} led at {LEAD.return_pct}% as of {end:%B} {end.day + 1}, {end.year}."
    assert llm.number_check(good, FACTS) == []
    assert llm.number_check(bad, FACTS) != []


@pytest.mark.parametrize("prose", [
    "XOM rose 999.99% in the period.",                       # invented return
    "Energy beat the market by 12.34 points.",               # derived number
    "Prices as of 2026-01-01.",                               # wrong date
])
def test_hallucinated_numbers_fall_back_to_template(prose):
    api = FakeAPI((200, tool(PLAN)), (200, text(prose)))
    out = answer("q", CON, client=api.client())
    assert out["writer"] == "template (model draft failed number check)" and out["rejected_numbers"]
    assert out["text"] == llm.write_template(out["facts"])


def test_bad_request_retries_plain():
    api = FakeAPI((400, {"type": "error", "error": {"type": "invalid_request_error", "message": "fallbacks"}}),
                  (200, tool(PLAN)),
                  (400, {"type": "error", "error": {"type": "invalid_request_error", "message": "fallbacks"}}),
                  (200, text(f"{LEAD.ticker} led with {LEAD.return_pct}%.")))
    out = answer("q", CON, client=api.client())
    assert out["parser"] == "claude-opus-5-5" and out["writer"] == "claude-opus-5-5"
    retry = api.requests[1]["body"]
    assert "fallbacks" not in retry and "strict" not in retry["tools"][0]


@pytest.mark.parametrize("status,err", [(401, "authentication_error"), (429, "rate_limit_error"),
                                        (500, "api_error"), (529, "overloaded_error")])
def test_api_errors_fall_back_to_rules(status, err):
    body = {"type": "error", "error": {"type": err, "message": "x"}}
    api = FakeAPI((status, body), (status, body))
    out = answer("Top 3 oil and gas names over 3 months", CON, client=api.client())
    assert out["status"] == "answered"
    assert out["parser"].startswith("rules (model call failed") and out["writer"].startswith("template (model call failed")


def test_model_refusal_and_missing_tool_call():
    api = FakeAPI((200, msg([], "refusal")), (200, text("ok")))
    assert answer("top 3 banks", CON, client=api.client())["parser"] == "rules (model returned no plan)"
    api = FakeAPI((200, text("I think you mean banks")), (200, text("ok")))
    assert answer("top 3 banks", CON, client=api.client())["parser"] == "rules (model returned no plan)"


@pytest.mark.parametrize("q", ["Top banks over the last 2 years", "Best tech stocks over the last 10 days"])
def test_unsupported_window_is_refused_before_the_model(q):
    api = FakeAPI()  # any model call would fail: no responses queued
    out = answer(q, CON, client=api.client())
    assert out["status"] == "refused_hard" and out["parser"] == "rules (window check)" and not api.requests


def test_model_plan_is_still_validated():
    for plan in [{"intent": "unsupported", "industries": [], "top_n": 10, "window": "3M", "reason": "No crypto data."},
                 {"intent": "top_performers", "industries": ["Banks"], "top_n": 500, "window": "3M"}]:
        out = answer("q", CON, client=FakeAPI((200, tool(plan))).client())
        assert out["status"] == "refused_hard"


# ---------------------------------------------------------------- app (headless Streamlit)

class FakeClient:
    """Stands in for the SDK client inside the app: plans from the rule parser, prose echoes a real fact."""

    def __init__(self):
        self.beta = NS(messages=NS(create=self.create))
        self.messages = self.beta.messages

    def create(self, **kw):
        if "tools" in kw:
            plan = llm.parse_rules(kw["messages"][0]["content"], INDS)
            return NS(model="claude-opus-5-5", stop_reason="tool_use", content=[NS(type="tool_use", input=plan)])
        facts = json.loads(kw["messages"][0]["content"].split("FACTS:\n", 1)[1])
        r = facts["results"][0]
        return NS(model="claude-opus-5-5", stop_reason="end_turn",
                  content=[NS(type="text", text=f"{r['ticker']} led, ending at ${r['end_close']} ({r['return_pct']}%).")])


def app(secrets=None, model=False, monkeypatch=None):
    if model:
        monkeypatch.setattr(llm, "make_client", lambda key=None: FakeClient() if key else None)
    at = AppTest.from_file("app.py", default_timeout=60)
    for k, v in (secrets or {}).items():
        at.secrets[k] = v
    return at.run()


def ask(at, q):
    at.text_input[0].set_value(q)
    next(b for b in at.button if b.label == "Ask").click()
    return at.run()


def texts(at):
    return " ".join(m.value for m in at.markdown)


def test_app_opens_locally_without_password():
    at = app()
    assert not at.exception and [t.label for t in at.tabs] == ["Ask", "Test set", "Audit log", "Method"]
    assert "rules and templates" in texts(at)


def test_hosted_copy_without_password_is_locked(monkeypatch):
    monkeypatch.setenv("REQUIRE_PASSWORD", "1")
    at = app()
    assert "isn't open yet" in texts(at) and not at.tabs


def test_password_gate(monkeypatch):
    monkeypatch.setenv("REQUIRE_PASSWORD", "1")
    at = app({"APP_PASSWORD": "test-only-pw"})
    assert not at.tabs and "password from my email" in texts(at)
    at.text_input[0].set_value("nope"); at.button[0].click(); at.run()
    assert at.error[0].value == "That password doesn't match." and not at.tabs
    at.text_input[0].set_value("test-only-pw"); at.button[0].click(); at.run(); at.run()
    assert [t.label for t in at.tabs][0] == "Ask"


def test_ask_answer_followup_and_refusal():
    at = app()
    at = ask(at, "Top 5 banks versus top 5 tech over 6 months")
    assert not at.exception
    body = texts(at)
    assert "Technology came out ahead" in body and "Read as: Top 5 in each" in body and "Numbers match" in body
    follow = [b for b in at.button if b.label.startswith("Compare top 5")][0]
    follow.click(); at.run()
    newest_q, newest = at.session_state["history"][0]
    assert newest_q == follow.label and newest["plan"]["window"] == "1Y" and newest["status"] == "answered"
    at = ask(at, "Which tech stock should I buy?")
    assert at.session_state["history"][0][1]["status"] == "refused_hard"
    assert "give investment advice" in texts(at) and "no query was run" in texts(at)
    assert len(at.session_state["history"]) == 3


def test_html_in_question_is_escaped():
    at = ask(app(), "<script>alert(1)</script> banks")
    body = texts(at)
    assert "<script>" not in body and "&lt;script&gt;" in body


def test_empty_question_does_nothing():
    at = ask(app(), "   ")
    assert "Pick an example" in texts(at)


def test_test_set_tab_all_pass():
    at = app()
    next(b for b in at.button if b.label == "Run the test set").click()
    at.run()
    import evaluate
    df = at.dataframe[0].value
    assert len(df) == len(evaluate.CASES) and (df.result == "Pass").all()


def test_model_mode_in_app(monkeypatch):
    at = app({"ANTHROPIC_API_KEY": "sk-test"}, model=True, monkeypatch=monkeypatch)
    assert "Claude (claude-opus-5-5)" in texts(at)
    at = ask(at, "Top 3 oil and gas names over the last 3 months")
    body = texts(at)
    assert not at.exception and "\\$" in body and "Parsed by claude-opus-5-5" in body


def test_session_question_limit(monkeypatch):
    at = app({"ANTHROPIC_API_KEY": "sk-test"}, model=True, monkeypatch=monkeypatch)
    at.session_state["asked"] = 40
    at = ask(at, "top 3 banks")
    assert "limit of 40 questions" in at.info[0].value


def test_audit_log_shows_only_this_session():
    a = ask(app(), "top 3 banks")
    b = app()
    a.tabs[2]; b.tabs[2]
    assert "Nothing yet" in " ".join(c.value for c in b.caption)
