"""Language layer: Claude reads the question and writes the answer. It never does math.

Step 1  parse_question : free text  -> small query plan (forced tool call, fixed schema)
Step 2  write_answer   : validated facts -> short narrative
Step 3  number_check   : every number in the narrative must exist in the facts,
                         otherwise the draft is thrown away and a template is used.

If no API key is available (or the model call fails), a rule-based parser and a template
writer are used instead, so the whole pipeline still runs end to end.
"""
import json
import logging
import os
import re
from datetime import datetime

from engine import INTENTS, WINDOW_LABEL, WINDOWS

log = logging.getLogger("portfolio_qa")

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5-5")
# On a safety-classifier decline, the API re-runs the request on Anthropic's recommended fallback model.
BETAS = ["server-side-fallback-2026-07-01"]


def make_client(api_key=None):
    """Returns an Anthropic client, or None when no key is available (rules + template mode)."""
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    import anthropic
    return anthropic.Anthropic(api_key=key)


def llm_available(api_key=None):
    return bool(api_key or os.environ.get("ANTHROPIC_API_KEY"))


def _call(client, **kw):
    return client.beta.messages.create(model=MODEL, betas=BETAS, fallbacks="default",
                                       output_config={"effort": "low"}, **kw)


# ---------------------------------------------------------------- step 1: parse

PARSE_SYSTEM = """You turn a finance user's question into a query plan by calling the make_query_plan tool.
Always call the tool exactly once, even for questions you would refuse.
You do not answer the question and you do not write SQL.
Rules:
- Use only the industries listed in the tool schema. Map obvious synonyms (energy -> Oil & Gas, cars -> Automobiles).
- If the question is about an industry, asset class or metric that is not available, asks for a
  forecast, or asks for investment advice, set intent to "unsupported" and give a one sentence reason.
- Defaults when not stated: top_n = 10, window = 3M.
- "trending" or "over time" means intent "trend". Two or more industries with compare/versus means "compare_industries"."""


def _plan_tool(industry_list):
    return {
        "name": "make_query_plan",
        "description": "Structured plan for a governed portfolio performance query.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "intent": {"type": "string", "enum": INTENTS},
                "industries": {"type": "array", "items": {"type": "string", "enum": industry_list}},
                "top_n": {"type": "integer", "description": "Number of names per industry. Range checked downstream."},
                "window": {"type": "string", "enum": list(WINDOWS)},
                "reason": {"type": "string", "description": "Only when intent is unsupported."},
            },
            "required": ["intent", "industries", "top_n", "window"],
            "additionalProperties": False,
        },
    }


# Regexes, matched on word boundaries so "auto" does not match "automatically".
SYNONYMS = {
    "Oil & Gas": r"\b(oil|gas|energy|petroleum|drillers?|refiners?)\b",
    "Automobiles": r"\b(autos?|automobiles?|automakers?|car ?makers?|cars?|vehicles?|evs?|electric vehicles?)\b",
    "Banks": r"\b(banks?|banking|lenders?|financials?)\b",
    "Technology": r"\b(tech|technology|software|semis?|semiconductors?|chips?|chipmakers?)\b",
    "Healthcare": r"\b(health ?care|health|pharma|pharmaceuticals?|drugmakers?|biotech)\b",
}
ADVICE = r"\b(should i|will|predict|predictions?|forecasts?|next (week|month|quarter|year)|recommend\w*|buy|sell|invest in)\b"
SUPPORTED_SPANS = {("month", 1), ("month", 3), ("month", 6), ("month", 12), ("year", 1)}
WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "nine": 9, "twelve": 12}


def _unsupported(reason):
    return {"intent": "unsupported", "industries": [], "top_n": 10, "window": "3M", "reason": reason}


def parse_rules(question, industry_list):
    """Tier 0 fallback parser. Plain keyword rules, no model."""
    q = " " + question.lower().replace("-", " ") + " "
    if re.search(ADVICE, q):
        return _unsupported("I can report past performance from the data, but I can't forecast or give investment advice.")
    found = [i for i in industry_list if re.search(SYNONYMS.get(i, rf"\b{re.escape(i.lower())}\b"), q)]
    if re.search(r"\b(all|every|each) (industr|sector)", q):
        found = list(industry_list)
    if not found:
        return _unsupported(f"I only have data for these industries: {', '.join(industry_list)}.")
    # an explicit span we don't carry ("last 2 years", "10 days") is refused, not silently defaulted
    for n, unit in re.findall(r"\b(\d+|" + "|".join(WORDS) + r")\s*(day|week|month|year)s?\b", q):
        n = WORDS.get(n) or int(n)
        if (unit, n) not in SUPPORTED_SPANS:
            return _unsupported(f"Supported time windows are: {', '.join(WINDOW_LABEL.values())}.")
    m = re.search(r"\b(?:top|best|leading)\s+(\d+)", q)
    top_n = int(m.group(1)) if m else 10
    if re.search(r"\bytd\b|year to date|this year|since january", q):
        window = "YTD"
    elif re.search(r"\b(6|six) months?|half (a )?year", q):
        window = "6M"
    elif re.search(r"\b(3|three) months?|quarter", q):
        window = "3M"
    elif re.search(r"\b(12|twelve) months?|\b(1|one) year|past year|last year|annual|\byear\b", q):
        window = "1Y"
    elif re.search(r"\bmonth\b|monthly|\b(1|one) month", q):
        window = "1M"
    else:
        window = "3M"
    if len(found) >= 2:
        intent = "compare_industries"
    elif re.search(r"trend|over time", q):
        intent = "trend"
    else:
        intent = "top_performers"
    return {"intent": intent, "industries": found, "top_n": top_n, "window": window}


def parse_question(question, industry_list, client=None):
    """Returns (plan, parser). Falls back to the rule parser if the model is unavailable or returns no plan."""
    if client is None:
        return parse_rules(question, industry_list), "rules"
    try:
        msg = _call(client, max_tokens=4000, system=PARSE_SYSTEM,
                    tools=[_plan_tool(industry_list)],
                    messages=[{"role": "user", "content": question}])
    except Exception as e:  # network, auth, rate limit: keep the demo answering
        log.warning("parse call failed: %r", e)
        return parse_rules(question, industry_list), f"rules (model call failed: {type(e).__name__})"
    plan = next((b.input for b in msg.content if b.type == "tool_use"), None)
    if msg.stop_reason == "refusal" or not isinstance(plan, dict):
        return parse_rules(question, industry_list), "rules (model returned no plan)"
    return plan, msg.model


# ---------------------------------------------------------------- step 2: write

WRITE_SYSTEM = """You write a short answer for a finance user from verified query results, in the voice of a
careful analyst writing a note to a colleague.
Rules:
- Answer the question in the first sentence. Do not restate the question.
- Use ONLY the numbers in the FACTS block. Never calculate, round differently, estimate or add outside knowledge.
  That includes differences, ratios and counts that are not in the facts.
- Quote returns exactly as given, with a percent sign.
- Mention the time window. Write dates exactly as given (YYYY-MM-DD).
- If any tickers were excluded by data quality checks, say which and why in one sentence.
- No investment advice, no predictions, no reasons for price moves that are not in the facts.
- Two to four sentences of plain prose. No headings, bullets, bold or exclamation marks."""


def facts_block(facts):
    return json.dumps({
        "plan": facts["plan"],
        "window": WINDOW_LABEL[facts["plan"]["window"]],
        "window_start": facts["window_start"],
        "as_of": facts["as_of"],
        "metric_definition": facts["metric_definition"],
        "results": facts["table"].to_dict("records"),
        "industry_summary": facts["summary"].to_dict("records"),
        "data_quality_checks": facts["checks"],
    }, indent=1)


WINDOW_PHRASE = {"1M": "over the last month", "3M": "over the last 3 months", "6M": "over the last 6 months",
                 "1Y": "over the past year", "YTD": "so far this year"}


def _and(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _move(v):
    return f"up {v}%" if v >= 0 else f"down {abs(v)}%"


def write_template(facts):
    """Deterministic narrative. Used with no model, and as the fallback when a draft fails the number check."""
    p, table, summary = facts["plan"], facts["table"], facts["summary"]
    when = WINDOW_PHRASE[p["window"]]
    parts = []
    if len(summary) == 1:
        s = summary.iloc[0]
        rows = table[table.industry == s.industry]
        lead = rows.iloc[0]
        parts.append(f"{lead.ticker} led {s.industry} {when}, {_move(lead.return_pct)}.")
        if len(rows) > 1:
            nxt = " and ".join(f"{r.ticker} ({r.return_pct}%)" for r in rows.iloc[1:3].itertuples())
            parts.append(f"Next were {nxt}. Across the top {int(s.tickers)}, the average return was {s.avg_return_pct}%.")
    else:
        ranked = summary.sort_values("avg_return_pct", ascending=False)
        best, rest = ranked.iloc[0], ranked.iloc[1:]
        others = _and(f"{r.industry} ({r.avg_return_pct}%)" for r in rest.itertuples())
        parts.append(f"{best.industry} came out ahead {when}: its top {int(best.tickers)} averaged "
                     f"{best.avg_return_pct}%, against {others}.")
        parts.append("The leaders were " + _and(
            f"{r.best_ticker} in {r.industry} ({r.best_return_pct}%)" for r in ranked.itertuples()) + ".")
    for c in facts["checks"]:
        if c["status"] == "excluded":
            parts.append(f"{_and(c['tickers'])} {'was' if len(c['tickers']) == 1 else 'were'} left out: "
                         f"{c['detail'][0].lower() + c['detail'][1:]}")
    parts.append(f"Prices from {facts['window_start']} to {facts['as_of']}.")
    return " ".join(parts)


# ---------------------------------------------------------------- step 3: verify

def allowed_numbers(facts):
    nums = set()
    for df in (facts["table"], facts["summary"]):
        for v in df.select_dtypes("number").to_numpy().ravel():
            nums.add(round(float(v), 2))
    nums.update(float(x) for x in (facts["plan"]["top_n"], 1, 3, 6, 12, len(facts["table"]),
                                   len(facts["summary"]), len(facts["plan"]["industries"])))
    return nums


def number_check(text, facts):
    """Post-flight guardrail. Returns the list of numbers in the text that are not in the facts."""
    ok_dates = {facts["window_start"], facts["as_of"]}
    bad = [d for d in re.findall(r"\d{4}-\d{2}-\d{2}", text) if d not in ok_dates]
    clean = re.sub(r"\d{4}-\d{2}-\d{2}", " ", text)
    for m in re.finditer(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? (\d{1,2}),? (\d{4})", clean):
        try:
            d = datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%b %d %Y").date().isoformat()
        except ValueError:
            d = None
        if d not in ok_dates:
            bad.append(m.group(0))
    clean = re.sub(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? \d{1,2},? \d{4}", " ", clean)
    allowed = allowed_numbers(facts)
    for tok in re.findall(r"(?<![A-Za-z])-?\d+(?:\.\d+)?", clean.replace(",", "")):
        v = abs(float(tok))
        if not any(abs(v - abs(a)) <= 0.051 for a in allowed):
            bad.append(tok)
    return bad


def write_answer(question, facts, client=None):
    """Returns (narrative, source, rejected_numbers)."""
    if client is None:
        return write_template(facts), "template", []
    try:
        msg = _call(client, max_tokens=4000, system=WRITE_SYSTEM,
                    messages=[{"role": "user", "content": f"QUESTION: {question}\n\nFACTS:\n{facts_block(facts)}"}])
    except Exception as e:
        log.warning("write call failed: %r", e)
        return write_template(facts), f"template (model call failed: {type(e).__name__})", []
    draft = "".join(b.text for b in msg.content if b.type == "text").strip()
    if msg.stop_reason == "refusal" or not draft:
        return write_template(facts), "template (model returned no draft)", []
    bad = number_check(draft, facts)
    if bad:
        return write_template(facts), "template (model draft failed number check)", bad
    return draft, msg.model, []
