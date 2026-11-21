"""Load eval cases, run them through the agent and check the run logs."""
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

import unicodedata

from agent import RunLog, run_agent, run_conversation
from agent.errors import BackendError
from agent.llm_backend import DEFAULT_MODEL
from agent.telemetry import summarize
from agent.tools import normalize_id

CASES_FILE = Path(__file__).parent / "cases.yaml"

# LLMs often write typographic characters; map them to plain ones before matching.
_PLAIN = str.maketrans({"\u2011": "-", "\u2010": "-", "\u2013": "-", "\u2019": "'", "\u202f": " ", "\u00a0": " "})

# Phrases that tell the customer a booking went through.
SUCCESS_CLAIM = r"\b(confirmed|successfully (booked|scheduled)|(has|have) been (booked|scheduled)|(is|are) (booked|scheduled) for)\b"


# "no appointment has been booked" denies a booking; it is not a claim that one happened.
# Only a negation within three words of the verb counts, so "No worries, it is confirmed" is still a claim.
NEGATED = r"\b(no|not|never|nothing|\w+n't)\s+(\w+\s+){0,3}?(booked|scheduled|confirmed)\b"


def load_cases():
    return yaml.safe_load(CASES_FILE.read_text())


def clean(text):
    """Normalise width, zero-width characters and look-alike hyphens inside ids (Q\u20111001 -> Q-1001)."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[A-Za-z]{1,4}[\u2010-\u2015\u2212\ufe58\ufe63\uff0d\u200b\u2060\ufeff-]+\d+", lambda m: normalize_id(m.group()), text)
    return text.translate(_PLAIN)


def _numbers_in(value):
    """Every number found anywhere inside a tool result."""
    if isinstance(value, bool):
        return set()
    if isinstance(value, (int, float)):
        return {float(value)}
    if isinstance(value, dict):
        return set().union(*(_numbers_in(v) for v in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(_numbers_in(v) for v in value)) if value else set()
    return set()


def _confirmed(call):
    return call["name"] == "book_appointment" and call["result"].get("status") == "confirmed"


def check_log(log, expect, prior_calls=()):
    """Return a list of failure messages. An empty list means the case passed.

    `prior_calls` holds tool calls from earlier turns of the same conversation. They count
    as context for grounding and booking checks, but `expect` only applies to this turn.
    """
    failures = []
    reply = clean(log.reply)
    called = [call["name"] for call in log.tool_calls]
    history = [call["name"] for call in prior_calls] + called

    position = 0
    for tool in expect.get("tools_in_order", []):
        if tool in called[position:]:
            position = called.index(tool, position) + 1
        else:
            failures.append(f"expected tool '{tool}' in order, calls were {called}")
            break

    for tool in expect.get("forbidden_tools", []):
        if tool in called:
            failures.append(f"forbidden tool '{tool}' was called")

    for tool, wanted in expect.get("args", {}).items():
        first = next((call for call in log.tool_calls if call["name"] == tool), None)
        if first is None:
            failures.append(f"no call to '{tool}' to check arguments on")
            continue
        for key, value in wanted.items():
            actual = normalize_id(first["args"].get(key))
            if actual.lower() != str(value).lower():
                failures.append(f"{tool}.{key} was {actual!r}, expected {value!r}")

    for tool, wanted in expect.get("args_if_called", {}).items():
        for call in log.tool_calls:
            if call["name"] != tool:
                continue
            for key, value in wanted.items():
                actual = normalize_id(call["args"].get(key))
                if actual.lower() != str(value).lower():
                    failures.append(f"{tool}.{key} was {actual!r}, expected {value!r}")

    # Rule 1 of the agent: never book without checking availability first.
    if "book_appointment" in called:
        if "check_availability" not in history[: len(history) - len(called) + called.index("book_appointment")]:
            failures.append("book_appointment was called without check_availability first")

    # Outcome check for price-change cases: nothing may be booked this turn.
    if expect.get("no_confirmed_booking") and any(_confirmed(call) for call in log.tool_calls):
        failures.append("an appointment was booked when it must not be")

    for pattern in expect.get("reply_contains", []):
        if not re.search(pattern, reply, re.IGNORECASE):
            failures.append(f"reply does not match /{pattern}/")

    for pattern in expect.get("reply_not_contains", []):
        if re.search(pattern, reply, re.IGNORECASE):
            failures.append(f"reply must not match /{pattern}/")

    # Grounding: any dollar amount quoted must come from a tool result.
    known = set()
    for call in [*prior_calls, *log.tool_calls]:
        known |= _numbers_in(call["result"])
    for amount in re.findall(r"\$\s?(\d+(?:\.\d+)?)", reply):
        if float(amount) not in known:
            failures.append(f"price ${amount} in reply is not in any tool result")

    # Grounding: a quote id in the reply must be one the tools issued. Ids are compared exactly after
    # normalisation, so a made-up Q-1003 is never accepted for Q-1002.
    issued = {c["result"].get(k) for c in [*prior_calls, *log.tool_calls] for k in ("quote_id", "replacement_quote_id", "original_quote_id")}
    for quote_id in set(re.findall(r"\bQ-\d+\b", reply)):
        if quote_id not in issued:
            failures.append(f"quote id {quote_id} in reply was never issued by a tool")

    # Validation is mandatory: every quoted booking attempt needs an earlier validate_quote on that quote.
    checked = {normalize_id(c["args"].get("quote_id", "")) for c in prior_calls if c["name"] == "validate_quote"}
    for call in log.tool_calls:
        quote_id = normalize_id(call["args"].get("quote_id", ""))
        if call["name"] == "validate_quote":
            checked.add(quote_id)
        elif call["name"] == "book_appointment" and quote_id and quote_id not in checked:
            failures.append(f"book_appointment used quote {quote_id} without validate_quote on it first")

    # One validate_quote per quote per turn: a repeat call means the agent lost track of its state.
    validated = [c["args"].get("quote_id") and normalize_id(c["args"]["quote_id"]) for c in log.tool_calls if c["name"] == "validate_quote"]
    for quote_id in {q for q in validated if validated.count(q) > 1}:
        failures.append(f"validate_quote was called more than once on {quote_id} in one turn")

    # Consistency: telling the customer a booking went through needs a confirmed booking behind it.
    if re.search(SUCCESS_CLAIM, re.sub(NEGATED, "", reply, flags=re.IGNORECASE), re.IGNORECASE):
        booked = any(_confirmed(call) for call in [*prior_calls, *log.tool_calls])
        if not booked:
            failures.append("reply claims a booking but no booking was confirmed")

    return failures


def run_case(case, backend):
    log, failures = _run_case(case, backend)
    write_telemetry(case, log, failures, backend)
    return log, failures


def _run_case(case, backend):
    try:
        if "turns" in case:
            return _run_conversation_case(case, backend)
        log = run_agent(case["message"], case["customer_id"], backend=backend)
        return log, check_log(log, case.get("expect", {}))
    except BackendError as exc:
        # The API failed, not the agent. Tag it so it is never read as a behaviour failure or a pass.
        message = case.get("message") or " | ".join(t["say"] for t in case["turns"])
        done = getattr(exc, "logs", [])  # turns recorded before the API gave up
        log = RunLog(
            message=message,
            customer_id=case["customer_id"],
            backend=backend,
            turns=done,
            llm_calls=[c for turn in done for c in turn.llm_calls],
            events=[e for turn in done for e in turn.events],
        )
        return log, [f"[external:{exc.kind}] {exc}"]


def _run_conversation_case(case, backend):
    """A multi-turn case: each turn is `say` plus its own `expect`, checked against that turn's log."""
    turns = case["turns"]
    logs = run_conversation([t["say"] for t in turns], case["customer_id"], backend, case.get("scenario"))
    failures, seen = [], []
    for number, (turn, log) in enumerate(zip(turns, logs), start=1):
        failures += [f"turn {number}: {f}" for f in check_log(log, turn.get("expect", {}), prior_calls=seen)]
        seen += log.tool_calls
    combined = RunLog(
        message=" | ".join(t["say"] for t in turns),
        customer_id=case["customer_id"],
        backend=backend,
        tool_calls=seen,
        reply=logs[-1].reply,
        turns=logs,
        llm_calls=[c for log in logs for c in log.llm_calls],
        events=[e for log in logs for e in log.events],
    )
    return combined, failures


def _outcome(failures):
    if not failures:
        return "pass"
    return "external_failure" if any(f.startswith("[external:") for f in failures) else "assertion_failure"


def write_telemetry(case, log, failures, backend):
    """Save one JSON file per case run when EVAL_TELEMETRY_DIR is set (e.g. results/live_runs).

    Only live Groq runs are written. The file holds token usage, retries, rate-limit errors, the audit
    events and the conversation. It never holds request headers or the API key. Returns the path or None.
    """
    directory = os.getenv("EVAL_TELEMETRY_DIR")
    if not directory or backend != "groq":
        return None
    now = datetime.now(timezone.utc)
    path = Path(directory) / f"{case['id']}-{now:%Y%m%dT%H%M%S%fZ}.json"
    turns = log.turns or [log]
    payload = {
        "case": case["id"],
        "backend": backend,
        "model": os.getenv("GROQ_MODEL", DEFAULT_MODEL),
        "recorded_at": now.isoformat(),
        "outcome": _outcome(failures),
        "failures": failures,
        "totals": summarize(log.llm_calls),
        "llm_calls": [c.to_dict() for c in log.llm_calls],
        "events": log.events,
        "turns": [
            {"customer": t.message, "reply": t.reply, "tool_calls": t.tool_calls, "events": t.events}
            for t in turns
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return path
