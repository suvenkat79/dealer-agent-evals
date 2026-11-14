"""Load eval cases, run them through the agent and check the run logs."""
import re
from pathlib import Path

import yaml

from agent import run_agent

CASES_FILE = Path(__file__).parent / "cases.yaml"

# LLMs often write typographic characters; map them to plain ones before matching.
_PLAIN = str.maketrans({"\u2011": "-", "\u2010": "-", "\u2013": "-", "\u2019": "'", "\u202f": " ", "\u00a0": " "})

# Phrases that tell the customer a booking went through.
SUCCESS_CLAIM = r"\b(confirmed|successfully (booked|scheduled)|(has|have) been (booked|scheduled)|(is|are) (booked|scheduled) for)\b"


def load_cases():
    return yaml.safe_load(CASES_FILE.read_text())


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


def check_log(log, expect):
    """Return a list of failure messages. An empty list means the case passed."""
    failures = []
    reply = log.reply.translate(_PLAIN)
    called = [call["name"] for call in log.tool_calls]

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
            actual = first["args"].get(key)
            if str(actual).lower() != str(value).lower():
                failures.append(f"{tool}.{key} was {actual!r}, expected {value!r}")

    for tool, wanted in expect.get("args_if_called", {}).items():
        for call in log.tool_calls:
            if call["name"] != tool:
                continue
            for key, value in wanted.items():
                actual = call["args"].get(key)
                if str(actual).lower() != str(value).lower():
                    failures.append(f"{tool}.{key} was {actual!r}, expected {value!r}")

    # Rule 1 of the agent: never book without checking availability first.
    if "book_appointment" in called:
        if "check_availability" not in called[: called.index("book_appointment")]:
            failures.append("book_appointment was called without check_availability first")

    for pattern in expect.get("reply_contains", []):
        if not re.search(pattern, reply, re.IGNORECASE):
            failures.append(f"reply does not match /{pattern}/")

    for pattern in expect.get("reply_not_contains", []):
        if re.search(pattern, reply, re.IGNORECASE):
            failures.append(f"reply must not match /{pattern}/")

    # Grounding: any dollar amount quoted must come from a tool result.
    known = set()
    for call in log.tool_calls:
        known |= _numbers_in(call["result"])
    for amount in re.findall(r"\$\s?(\d+(?:\.\d+)?)", reply):
        if float(amount) not in known:
            failures.append(f"price ${amount} in reply is not in any tool result")

    # Consistency: telling the customer a booking went through needs a confirmed booking behind it.
    if re.search(SUCCESS_CLAIM, reply, re.IGNORECASE):
        booked = any(
            call["name"] == "book_appointment" and call["result"].get("status") == "confirmed"
            for call in log.tool_calls
        )
        if not booked:
            failures.append("reply claims a booking but no booking was confirmed")

    return failures


def run_case(case, backend):
    log = run_agent(case["message"], case["customer_id"], backend=backend)
    return log, check_log(log, case.get("expect", {}))
