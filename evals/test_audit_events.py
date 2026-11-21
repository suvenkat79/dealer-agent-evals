"""The quote workflow leaves an append-only audit trail: shown, answered, validated, booked, refused."""
import json

from agent import run_conversation
from agent.tools import Tools
from evals.quote_helpers import SLOT, quote
from evals.test_quote_workflow import CHANGE, Chat, quoted_chat


def kinds(tools):
    return [e["event"] for e in tools.events]


def test_the_happy_path_is_recorded_in_order():
    chat = quoted_chat({})
    chat.customer("Yes, $79 is fine, go ahead.")
    chat.tools.validate_quote("Q-1001")
    chat.book("Q-1001", 79)
    assert kinds(chat.tools) == [
        "quote_issued", "quote_presented", "customer_accepted", "quote_validated", "booking_confirmed",
    ]
    events = chat.tools.events
    assert [e["seq"] for e in events] == [1, 2, 3, 4, 5]
    assert [e["turn"] for e in events] == [1, 1, 2, 2, 2]
    assert events[1] == {"seq": 2, "turn": 1, "event": "quote_presented", "quote_id": "Q-1001", "price_usd": 79, "reopened": False}
    assert events[2]["price_usd"] == 79 and events[4]["quote_id"] == "Q-1001" and events[4]["confirmation_id"] == "APT-0001"


def test_same_turn_validation_and_booking_are_recorded_as_refusals():
    chat = Chat({})
    chat.customer("Book an oil change Friday at 10am.")
    quote(chat.tools)
    chat.tools.validate_quote("Q-1001")
    chat.book("Q-1001", 79)
    refusals = [e for e in chat.tools.events if e["event"] == "action_refused"]
    assert [(e["action"], e["reason"]) for e in refusals] == [
        ("validate_quote", "quote_not_presented"), ("book_appointment", "validation_required"),
    ]
    assert "booking_confirmed" not in kinds(chat.tools)


def test_a_price_change_records_expiry_and_a_fresh_acceptance():
    chat = quoted_chat(CHANGE)
    chat.customer("Yes, $150 is fine.")
    chat.tools.validate_quote("Q-1001")
    chat.agent_says("Quote Q-1001 ($150) expired. Quote Q-1002 is $180 (+$30). Do you want it?")
    chat.customer("Yes, $180 is fine. Book it.")
    chat.tools.validate_quote("Q-1002")
    chat.book("Q-1002", 180)
    assert kinds(chat.tools) == [
        "quote_issued", "quote_presented", "customer_accepted", "quote_expired", "quote_presented",
        "customer_accepted", "quote_validated", "booking_confirmed",
    ]
    expired = chat.tools.events[3]
    assert (expired["quoted_price_usd"], expired["current_price_usd"], expired["replacement_quote_id"]) == (150, 180, "Q-1002")
    assert [e["quote_id"] for e in chat.tools.events if e["event"] == "customer_accepted"] == ["Q-1001", "Q-1002"]


def test_an_unclear_or_declined_answer_is_recorded_and_never_as_acceptance():
    chat = quoted_chat({})
    chat.customer("ok maybe")
    chat.customer("No thanks.")
    assert kinds(chat.tools)[-2:] == ["customer_response_unclear", "customer_rejected"]
    assert "customer_accepted" not in kinds(chat.tools)


def test_events_hold_no_customer_text():
    chat = quoted_chat({})
    chat.customer("Yes, $79 is fine, go ahead. My secret word is pineapple.")
    assert "pineapple" not in json.dumps(chat.tools.events)


def test_the_trail_is_append_only_and_callers_get_copies():
    chat = quoted_chat({})
    copy = chat.tools.events_since(0)
    copy[0]["event"] = "tampered"
    copy.clear()
    assert chat.tools.events[0]["event"] == "quote_issued"
    before = len(chat.tools.events)
    chat.customer("Yes, $79 is fine.")
    assert len(chat.tools.events) > before and chat.tools.events[:before] == chat.tools.events_since(0)[:before]


def test_direct_tool_use_without_turns_is_audited_too():
    tools = Tools(scenario={}, customer_id="C001")
    quote_id = quote(tools)["quote_id"]
    tools.validate_quote(quote_id)
    tools.book_appointment(**SLOT, quote_id=quote_id)
    assert kinds(tools) == ["quote_issued", "quote_validated", "booking_confirmed"]
    assert all(e["turn"] is None for e in tools.events)


def test_refused_quotes_are_recorded_with_their_reason():
    tools = Tools(scenario={}, customer_id="C001")
    tools.get_quote("oil change", "Civic", SLOT["date"], SLOT["time"], "C001", "V001")
    assert tools.events[0] == {"seq": 1, "turn": None, "event": "action_refused", "action": "get_quote", "reason": "availability_not_checked"}


def test_each_turn_log_carries_its_own_events():
    logs = run_conversation(["Book an oil change Friday at 10am.", "Yes, $79 is fine."], "C001", "guarded", {})
    assert [e["event"] for e in logs[0].events] == ["quote_issued", "quote_presented"]
    assert [e["event"] for e in logs[1].events] == ["customer_accepted", "quote_validated", "booking_confirmed"]
    assert json.dumps([log.to_dict() for log in logs])  # serialisable
