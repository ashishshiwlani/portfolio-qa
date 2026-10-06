"""One function that runs the whole flow and writes an audit record."""
import hashlib
import json
import time
from datetime import datetime, timezone

import engine
import llm

AUDIT_FILE = "audit_log.jsonl"


def answer(question, con, log=True, client=None, session=None):
    t0 = time.perf_counter()
    inds = engine.industries(con)
    plan, parser = llm.parse_question(question, inds, client)
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "session": session, "question": question, "parser": parser, "plan": plan}
    try:
        facts = engine.run(plan, con)
        text, writer, rejected = llm.write_answer(question, facts, client)
        out = {"status": "answered", "text": text, "writer": writer, "rejected_numbers": rejected,
               "facts": facts, "plan": facts["plan"], "parser": parser}
        rec.update(status="answered", writer=writer, rejected_numbers=rejected, answer=text,
                   sql_params=facts["sql_params"], checks=facts["checks"],
                   result_hash=hashlib.sha256(facts["table"].to_csv(index=False).encode()).hexdigest()[:16])
    except engine.Refusal as r:
        out = {"status": f"refused_{r.kind}", "text": r.message, "plan": plan, "parser": parser}
        rec.update(status=out["status"], answer=r.message)
    out["latency_ms"] = rec["latency_ms"] = round((time.perf_counter() - t0) * 1000)
    if log:
        with open(AUDIT_FILE, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return out


def read_audit(session=None, limit=200):
    """Newest first. With a session id, only that session's requests."""
    try:
        with open(AUDIT_FILE) as f:
            rows = [json.loads(l) for l in f]
    except FileNotFoundError:
        return []
    if session:
        rows = [r for r in rows if r.get("session") == session]
    return rows[::-1][:limit]
