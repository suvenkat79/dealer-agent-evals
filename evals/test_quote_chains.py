"""F8: after a quote is declined, a different slot (or a fresh request) gets a fresh quote of its own.

An "original quote" is per chain now: a fresh quote is its own origin, a replacement stands in for the
quote it replaced. A rejected quote is never handed out again and never booked on its old acceptance.
"""
import pytest

from evals.quote_helpers import SLOT, quote
from evals.test_quote_workflow import CHANGE, Chat, quoted_chat

LATER = dict(SLOT, time="11:00")


def decline_first_quote(scenario):
    chat = quoted_chat(scenario)
    chat.customer("No thanks.")
    chat.agent_says("Understood, nothing is booked.")
    chat.customer("How about 11:00 instead?")
    return chat


def present_and_accept(chat, quote_id, price, slot_text="11:00"):
    chat.agent_says(f"Quote {quote_id} is ${price} for {slot_text}. Shall I book it?")
    chat.customer(f"Yes, ${price} is fine.")


@pytest.mark.parametrize("scenario, price", [({}, 79), (CHANGE, 180)])
def test_a_new_slot_after_a_decline_gets_a_fresh_bound_quote_that_can_be_booked(scenario, price):
    chat = decline_first_quote(scenario)
    fresh = quote(chat.tools, time="11:00")
    assert fresh["quote_id"] == "Q-1002" and fresh["price_usd"] == price and "error" not in fresh
    assert chat.tools.quotes["Q-1002"] == {
        "service_type": "oil change", "vehicle_model": "Civic", "price_usd": price,
        "customer_id": "C001", "vehicle_id": "V001", "date": "2026-11-20", "time": "11:00",
    }
    present_and_accept(chat, "Q-1002", price)
    assert "error" not in chat.tools.validate_quote("Q-1002")  # not blocked by the declined Q-1001
    booking = chat.tools.book_appointment(**LATER, quote_id="Q-1002", confirmed_price_usd=str(price))
    assert (booking["status"], booking["quote_id"], booking["time"], booking["price_usd"]) == ("confirmed", "Q-1002", "11:00", price)


@pytest.mark.parametrize("scenario", [{}, CHANGE])
def test_the_declined_quote_stays_declined_and_unbookable(scenario):
    chat = decline_first_quote(scenario)
    fresh = quote(chat.tools, time="11:00")
    present_and_accept(chat, fresh["quote_id"], fresh["price_usd"])
    chat.tools.validate_quote(fresh["quote_id"])
    assert chat.tools.quote_state()["quotes"]["Q-1001"]["customer_decision"] == "rejected"
    assert chat.tools.validate_quote("Q-1001")["reason"] == "customer_rejected"
    assert chat.book("Q-1001", 150 if scenario else 79)["status"] == "failed"
    assert chat.tools.db["slots"][SLOT["date"]][SLOT["time"]] == "free"


def test_quotes_for_different_slots_cannot_stand_in_for_each_other():
    chat = decline_first_quote({})
    fresh = quote(chat.tools, time="11:00")
    present_and_accept(chat, fresh["quote_id"], 79)
    chat.tools.validate_quote("Q-1002")
    assert chat.tools.book_appointment(**SLOT, quote_id="Q-1002")["reason"] == "quote_slot_mismatch"
    assert chat.tools.book_appointment(**LATER, quote_id="Q-1001")["status"] == "failed"
    assert chat.tools.db["slots"][SLOT["date"]]["10:00"] == "free" and chat.tools.db["slots"][SLOT["date"]]["11:00"] == "free"


def test_a_fresh_quote_is_its_own_origin():
    chat = decline_first_quote(CHANGE)
    quote(chat.tools, time="11:00")
    state = chat.tools.quote_state()["quotes"]
    assert (state["Q-1001"]["original_quote_id"], state["Q-1002"]["original_quote_id"]) == ("Q-1001", "Q-1002")
    present_and_accept(chat, "Q-1002", 180)
    check = chat.tools.validate_quote("Q-1002")
    assert (check["status"], check["original_quote_id"], check["original_price_usd"]) == ("valid", "Q-1002", 180)


def test_asking_again_for_the_declined_slot_never_hands_back_the_rejected_quote():
    chat = quoted_chat({})
    chat.customer("No thanks.")
    chat.agent_says("OK.")
    chat.customer("Actually, 10:00 again please.")
    again = quote(chat.tools)
    assert again["quote_id"] == "Q-1002"
    state = chat.tools.quote_state()["quotes"]
    assert state["Q-1001"]["customer_decision"] == "rejected"
    assert (state["Q-1002"]["presented_turn"], state["Q-1002"]["customer_decision"]) == (None, None)
    assert chat.tools.validate_quote("Q-1002")["reason"] == "quote_not_presented"  # fresh consent cycle


def test_the_same_slot_under_a_price_change_is_not_blocked_by_the_declined_expired_quote():
    chat = quoted_chat(CHANGE)  # Q-1001 at $150 is now expired
    chat.customer("No thanks.")
    chat.agent_says("OK.")
    chat.customer("Actually, 10:00 again.")
    again = quote(chat.tools)
    assert (again.get("quote_id"), again.get("price_usd")) == ("Q-1002", 180)


def test_an_undecided_expired_quote_still_blocks_a_new_quote_for_the_same_slot_only():
    chat = quoted_chat(CHANGE)  # Q-1001 at $150 expired, the customer has not answered
    assert "still outstanding" in quote(chat.tools)["error"]  # same slot: validate Q-1001 first
    other = quote(chat.tools, time="11:00")  # a different slot is independent
    assert (other["quote_id"], other["price_usd"]) == ("Q-1002", 180)


def test_the_replacement_chain_is_unchanged():
    chat = quoted_chat(CHANGE)
    chat.customer("Yes, $150 is fine.")
    check = chat.tools.validate_quote("Q-1001")
    assert (check["status"], check["replacement_quote_id"], check["original_quote_id"]) == ("expired", "Q-1002", "Q-1001")
    assert chat.tools.quote_state()["quotes"]["Q-1002"]["original_quote_id"] == "Q-1001"
    assert chat.tools.book_appointment(**SLOT, quote_id="Q-1002")["reason"] == "price_confirmation_required"


def test_a_withdrawn_acceptance_is_never_reused_after_the_quote_is_shown_again():
    chat = quoted_chat({})
    chat.customer("Yes, $79 is fine.")
    chat.tools.validate_quote("Q-1001")
    chat.agent_says("Shall I confirm the oil change at 10:00 for $79?")
    chat.customer("Actually no, cancel it.")
    assert chat.book("Q-1001", 79)["reason"] == "customer_rejected"
    chat.agent_says("Understood, nothing is booked.")
    chat.customer("Hmm, tell me the details again.")
    assert chat.book("Q-1001", 79)["reason"] == "customer_rejected"  # not shown again yet
    chat.agent_says("Quote Q-1001 is $79 for the oil change at 10:00. Shall I book it?")  # shown again
    assert chat.tools.quote_state()["quotes"]["Q-1001"]["customer_decision"] is None  # the old yes is gone
    assert chat.book("Q-1001", 79)["reason"] == "quote_not_presented"  # same turn it is shown again
    chat.customer("Hmm, let me think.")
    assert chat.book("Q-1001", 79)["reason"] == "customer_acceptance_required"  # an unclear answer is not a yes
    chat.customer("Yes, $79 is fine.")
    assert chat.book("Q-1001", 79)["status"] == "confirmed"


def test_a_rejected_quote_cannot_be_booked_while_a_fresh_one_exists():
    chat = quoted_chat({})
    chat.customer("Yes, $79 is fine.")
    chat.tools.validate_quote("Q-1001")
    chat.agent_says("Confirm the oil change at 10:00 for $79?")
    chat.customer("No, cancel it.")
    fresh = quote(chat.tools)
    assert fresh["quote_id"] == "Q-1002"
    assert chat.book("Q-1001", 79)["reason"] == "customer_rejected"
    assert chat.tools.db["slots"][SLOT["date"]][SLOT["time"]] == "free"


def test_fresh_quotes_are_audited_as_new_not_reused():
    chat = decline_first_quote({})
    quote(chat.tools, time="11:00")
    issued = [e for e in chat.tools.events if e["event"] == "quote_issued"]
    assert [(e["quote_id"], e["time"], e["reused"]) for e in issued] == [("Q-1001", "10:00", False), ("Q-1002", "11:00", False)]
