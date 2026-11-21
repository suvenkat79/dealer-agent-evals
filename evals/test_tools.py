"""Deterministic tests of the quote tools. No agent and no API key involved."""
from agent.tools import Tools
from evals.quote_helpers import quote

CHANGE = {"price_change": {"service_type": "oil change", "original_usd": 150, "updated_usd": 180}}
BOOKING = dict(customer_id="C001", vehicle_id="V001", date="2026-11-20", time="10:00", service_type="oil change")


def test_price_moves_after_first_quote_and_expires_it():
    tools = Tools(scenario=CHANGE, customer_id="C001")
    issued = quote(tools)
    assert (issued["quote_id"], issued["price_usd"]) == ("Q-1001", 150)
    check = tools.validate_quote("Q-1001")
    assert check["status"] == "expired"
    assert (check["quoted_price_usd"], check["current_price_usd"], check["price_change_usd"]) == (150, 180, 30)


def test_expired_validation_returns_a_replacement_and_never_a_third_quote():
    tools = Tools(scenario=CHANGE, customer_id="C001")
    quote(tools)
    check = tools.validate_quote("Q-1001")
    assert check["replacement_quote_id"] == "Q-1002"
    assert quote(tools)["quote_id"] == "Q-1002"  # reused, not Q-1003
    assert tools.validate_quote("Q-1001")["replacement_quote_id"] == "Q-1002"
    assert set(tools.quotes) == {"Q-1001", "Q-1002"}


def test_booking_requires_validation():
    tools = Tools(scenario=CHANGE, customer_id="C001")
    quote(tools)
    assert tools.book_appointment(**BOOKING, quote_id="Q-1001")["reason"] == "validation_required"


def test_expired_quote_is_never_bookable():
    tools = Tools(scenario=CHANGE, customer_id="C001")
    quote(tools)
    tools.validate_quote("Q-1001")  # reports expired; does not mark the old quote validated
    result = tools.book_appointment(**BOOKING, quote_id="Q-1001", confirmed_price_usd="150")
    assert (result["status"], result["reason"]) == ("failed", "validation_required")
    assert "Q-1001" not in tools.validated


def test_replacement_quote_books_at_the_new_price():
    tools = Tools(scenario=CHANGE, customer_id="C001")
    quote(tools)
    tools.validate_quote("Q-1001")
    result = tools.book_appointment(**BOOKING, quote_id="Q-1002", confirmed_price_usd="180")
    assert (result["status"], result["price_usd"], result["quote_id"]) == ("confirmed", 180, "Q-1002")


def test_wrong_confirmed_price_is_rejected():
    tools = Tools(scenario=CHANGE, customer_id="C001")
    quote(tools)
    tools.validate_quote("Q-1001")
    assert tools.book_appointment(**BOOKING, quote_id="Q-1002", confirmed_price_usd="150")["reason"] == "price_mismatch"


def test_unchanged_price_validates_and_books_on_the_original_quote():
    tools = Tools(scenario={}, customer_id="C001")
    assert quote(tools)["quote_id"] == "Q-1001"
    assert tools.validate_quote("Q-1001")["status"] == "valid"
    assert tools.book_appointment(**BOOKING, quote_id="Q-1001", confirmed_price_usd="79")["status"] == "confirmed"
    # The slot is now taken: it can no longer be quoted, and a quote for another free slot gets a new id.
    assert quote(tools)["reason"] == "slot_not_available"
    assert quote(tools, time="11:00")["quote_id"] == "Q-1002"
