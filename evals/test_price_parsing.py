"""F6: a malformed confirmed_price_usd gets a structured error, never an exception."""
from decimal import Decimal

import pytest

from agent.tools import parse_price
from evals.quote_helpers import SLOT, new_tools, quote
from evals.test_groq_backend import calls, groq, talk, text  # noqa: F401  (groq is a fixture)

VALID = [79, 79.0, 79.00, "79", "79.0", "79.00", "$79", " $79 ", "$79.00", "079"]
INVALID = [
    True, False, float("nan"), float("inf"), float("-inf"), -5, -0.01, "-5", "-$5", "$-5",
    "", " ", "$", "abc", "$79x", "79 dollars", "79 USD", "seventy-nine", "79.", ".79", "79.001", "79.5.0",
    "1e3", "0x4f", "1,07", "1,0790", ",079", "1,,079", "$ 79", "＄79", "٧٩", "79\n80",
    [], {}, (79,), 79.001, 1e30, 10**9, "9999999",
]


@pytest.mark.parametrize("value", VALID)
def test_clean_amounts_are_parsed(value):
    assert parse_price(value) == Decimal(79)


@pytest.mark.parametrize("value, expected", [("1,079", 1079), ("$1,079.50", Decimal("1079.50")), (1079, 1079), ("1079", 1079)])
def test_comma_grouping_is_accepted_only_in_threes(value, expected):
    assert parse_price(value) == Decimal(expected)


@pytest.mark.parametrize("value", INVALID)
def test_everything_else_is_rejected_by_the_parser(value):
    assert parse_price(value) is None


@pytest.mark.parametrize("value", INVALID)
def test_an_invalid_price_is_a_structured_refusal_not_an_exception(value):
    tools = new_tools(scenario={})
    quote_id = quote(tools)["quote_id"]
    tools.validate_quote(quote_id)
    result = tools.call("book_appointment", {**SLOT, "quote_id": quote_id, "confirmed_price_usd": value})
    assert (result["status"], result["reason"]) == ("failed", "invalid_confirmed_price")
    assert tools.db["slots"][SLOT["date"]][SLOT["time"]] == "free"
    refused = [e for e in tools.events if e["event"] == "action_refused" and e["action"] == "book_appointment"]
    assert refused[-1]["reason"] == "invalid_confirmed_price"


@pytest.mark.parametrize("value", ["$79", "79", "79.00", 79, 79.0])
def test_valid_formats_book(value):
    tools = new_tools(scenario={})
    quote_id = quote(tools)["quote_id"]
    tools.validate_quote(quote_id)
    assert tools.call("book_appointment", {**SLOT, "quote_id": quote_id, "confirmed_price_usd": value})["status"] == "confirmed"


def test_a_valid_but_wrong_price_is_still_a_price_mismatch():
    tools = new_tools(scenario={})
    quote_id = quote(tools)["quote_id"]
    tools.validate_quote(quote_id)
    result = tools.call("book_appointment", {**SLOT, "quote_id": quote_id, "confirmed_price_usd": "$60"})
    assert (result["reason"], result["current_price_usd"]) == ("price_mismatch", 79)


def test_a_dollar_sign_price_is_checked_against_the_current_price_not_trusted():
    change = {"price_change": {"service_type": "oil change", "original_usd": 150, "updated_usd": 180}}
    tools = new_tools(scenario=change)
    quote(tools)
    assert tools.validate_quote("Q-1001")["replacement_quote_id"] == "Q-1002"  # expired: replacement at $180
    result = tools.call("book_appointment", {**SLOT, "quote_id": "Q-1002", "confirmed_price_usd": "$150"})
    assert (result["reason"], result["current_price_usd"]) == ("price_mismatch", 180)
    assert tools.call("book_appointment", {**SLOT, "quote_id": "Q-1002", "confirmed_price_usd": "$180"})["status"] == "confirmed"


def test_a_model_that_sends_a_dollar_sign_or_garbage_does_not_abort_the_conversation(groq):
    groq(
        calls(("check_availability", {"date": "2026-11-20", "service_type": "oil change"}),
              ("get_quote", {"service_type": "oil change", "vehicle_model": "Civic", "date": "2026-11-20", "time": "10:00",
                             "customer_id": "C001", "vehicle_id": "V001"})),
        text("Quote Q-1001 is $79. Shall I book it?"),
        calls(("validate_quote", {"quote_id": "Q-1001"})),
        calls(("book_appointment", {**SLOT, "quote_id": "Q-1001", "confirmed_price_usd": "79 dollars"})),
        calls(("book_appointment", {**SLOT, "quote_id": "Q-1001", "confirmed_price_usd": "$79"})),
        text("Booked."),
    )
    logs = talk(["Book an oil change Friday at 10am.", "Yes, $79 is fine."], scenario={})
    results = [c["result"] for c in logs[1].tool_calls if c["name"] == "book_appointment"]
    assert results[0]["reason"] == "invalid_confirmed_price" and results[1]["status"] == "confirmed"
    assert logs[1].reply == "Booked."
