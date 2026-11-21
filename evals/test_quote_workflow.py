"""Server-side quote workflow: shown to the customer, accepted by them, validated, then booked.

These drive the tools the way the conversation runner does (begin_turn / end_turn), with no model.
Each test is one customer conversation; `say` plays one customer message and the agent's reply.
"""
import pytest

from agent import run_conversation
from agent.consent import classify_reply
from agent.tools import Tools
from evals.quote_helpers import quote

CHANGE = {"price_change": {"service_type": "oil change", "original_usd": 150, "updated_usd": 180}}
SLOT = dict(customer_id="C001", vehicle_id="V001", date="2026-11-20", time="10:00", service_type="oil change")
CAR = ("oil change", "Civic")


class Chat:
    """A conversation against the real tools. `turn` opens a customer turn; `reply` closes it."""

    def __init__(self, scenario):
        self.tools = Tools(scenario=scenario)
        self.number = 0

    def customer(self, message):
        self.number += 1
        self.tools.begin_turn(self.number, message)

    def agent_says(self, reply):
        self.tools.end_turn(reply)

    def book(self, quote_id, price):
        return self.tools.book_appointment(**SLOT, quote_id=quote_id, confirmed_price_usd=str(price))


def quoted_chat(scenario=CHANGE):
    chat = Chat(scenario)
    chat.customer("I want to book an oil change on Friday at 10am.")
    quoted = quote(chat.tools)
    chat.agent_says(f"Quote {quoted['quote_id']} is ${quoted['price_usd']}. Shall I book it?")
    return chat


# 1. The original quote is shown and booking waits for acceptance.
def test_nothing_can_be_validated_or_booked_in_the_turn_the_quote_is_first_shown():
    chat = Chat({})
    chat.customer("Book an oil change Friday 10am")
    quote(chat.tools)
    assert chat.tools.validate_quote("Q-1001")["reason"] == "quote_not_presented"
    assert chat.book("Q-1001", 79)["reason"] == "validation_required"  # validating was refused, so it never happened
    chat.agent_says("Quote Q-1001 is $79.")
    assert chat.tools.validate_quote("Q-1001")["reason"] == "quote_not_presented"  # still the same turn


def test_a_quote_the_reply_never_named_is_not_shown():
    chat = Chat({})
    chat.customer("Book an oil change Friday 10am")
    quote(chat.tools)
    chat.agent_says("It will be about $79.")  # no quote id
    chat.customer("Yes please")
    assert chat.tools.validate_quote("Q-1001")["reason"] == "quote_not_presented"


def test_booking_waits_for_a_reply_even_after_validation_was_attempted():
    chat = quoted_chat({})
    assert chat.book("Q-1001", 79)["reason"] == "validation_required"
    assert chat.tools.quote_state()["quotes"]["Q-1001"]["presented_turn"] == 1


# 2. The customer accepts an unchanged price: the original quote is validated, then booked.
def test_unchanged_price_accepted_then_validated_then_booked():
    chat = quoted_chat({})
    chat.customer("Yes, $79 is fine, go ahead.")
    assert chat.book("Q-1001", 79)["reason"] == "validation_required"  # accepted but not yet validated
    assert chat.tools.validate_quote("Q-1001")["status"] == "valid"
    booking = chat.book("Q-1001", 79)
    assert (booking["status"], booking["quote_id"], booking["price_usd"]) == ("confirmed", "Q-1001", 79)


# 3. The original expires and the replacement is shown.
def test_expired_original_gives_a_replacement_that_must_be_shown_before_it_can_be_used():
    chat = quoted_chat()
    chat.customer("Yes, $150 is fine.")
    check = chat.tools.validate_quote("Q-1001")
    assert (check["status"], check["replacement_quote_id"], check["price_change_usd"]) == ("expired", "Q-1002", 30)
    assert chat.book("Q-1002", 180)["reason"] == "quote_not_presented"  # same turn: customer has not seen it
    chat.agent_says("Quote Q-1001 ($150) is no longer valid. Quote Q-1002 is $180 (+$30). Do you want it?")
    assert chat.tools.quote_state()["quotes"]["Q-1002"]["presented_turn"] == 2


# 4. The customer accepts the replacement: it is validated, then booked.
def test_replacement_accepted_then_validated_then_booked():
    chat = quoted_chat()
    chat.customer("Yes, $150 is fine.")
    chat.tools.validate_quote("Q-1001")
    chat.agent_says("Quote Q-1001 ($150) expired. Quote Q-1002 is $180. Do you want it?")
    chat.customer("Yes, $180 is fine. Book it.")
    assert chat.book("Q-1002", 180)["reason"] == "validation_required"  # not validated since acceptance
    assert chat.tools.validate_quote("Q-1002")["status"] == "valid"
    booking = chat.book("Q-1002", 180)
    assert (booking["status"], booking["quote_id"], booking["price_usd"]) == ("confirmed", "Q-1002", 180)


def test_accepting_the_old_price_does_not_accept_the_new_one():
    chat = quoted_chat()
    chat.customer("Yes, $150 is fine.")
    chat.tools.validate_quote("Q-1001")
    chat.agent_says("Quote Q-1001 ($150) expired. Quote Q-1002 is $180. Do you want it?")
    chat.customer("Yes, $150 is fine, book it.")  # still answering the old price
    assert chat.tools.validate_quote("Q-1002")["reason"] == "customer_acceptance_required"
    assert chat.book("Q-1002", 180)["reason"] == "customer_acceptance_required"


# 5. The customer rejects the replacement: nothing can be booked.
def test_rejected_replacement_cannot_be_validated_or_booked_and_the_original_is_expired():
    chat = quoted_chat()
    chat.customer("Yes, $150 is fine.")
    chat.tools.validate_quote("Q-1001")
    chat.agent_says("Quote Q-1001 ($150) expired. Quote Q-1002 is $180. Do you want it?")
    chat.customer("No, I don't want to pay $180. Keep my original price.")
    assert chat.tools.validate_quote("Q-1002")["reason"] == "customer_rejected"
    assert chat.book("Q-1002", 180)["reason"] == "customer_rejected"
    assert chat.book("Q-1001", 150)["status"] == "failed"  # the expired original cannot be used instead
    assert chat.tools.db["slots"]["2026-11-20"]["10:00"] == "free"


def test_a_later_yes_after_a_rejection_needs_the_quote_to_be_shown_again():
    chat = quoted_chat()
    chat.customer("Yes, $150.")
    chat.tools.validate_quote("Q-1001")
    chat.agent_says("Q-1001 expired. Quote Q-1002 is $180. Want it?")
    chat.customer("No thanks")
    chat.agent_says("Understood, nothing is booked.")
    chat.customer("Actually, yes, $180 is fine.")
    assert chat.tools.validate_quote("Q-1002")["reason"] == "customer_rejected"
    chat.agent_says("Quote Q-1002 is $180. Shall I book it?")
    chat.customer("Yes, $180.")
    assert chat.tools.validate_quote("Q-1002")["status"] == "valid"


# 6. A model that asks to book without acceptance is blocked by the tools.
@pytest.mark.parametrize("answer", ["What does that include?", "Hmm, let me think", "Yes, but can you do $120?"])
def test_booking_without_a_clear_acceptance_is_blocked(answer):
    chat = quoted_chat({})
    chat.customer(answer)
    assert chat.tools.validate_quote("Q-1001")["reason"] == "customer_acceptance_required"
    assert chat.book("Q-1001", 79)["status"] == "failed"


def test_a_new_turn_alone_is_not_acceptance():
    chat = quoted_chat({})
    chat.customer("Can you tell me more about the oil change?")
    assert chat.book("Q-1001", 79)["reason"] in ("validation_required", "customer_acceptance_required")
    assert chat.tools.quote_state()["quotes"]["Q-1001"]["customer_decision"] is None


# 7. A model that asks to book without validation is blocked by the tools.
def test_booking_without_validation_is_blocked_even_after_acceptance():
    chat = quoted_chat({})
    chat.customer("Yes, $79 please.")
    result = chat.book("Q-1001", 79)
    assert result["status"] == "failed" and result["reason"] == "validation_required"
    assert "validate_quote" in result["detail"]


# Reading the customer's words
@pytest.mark.parametrize(
    "message, price, expected",
    [
        ("Yes, $150 is fine. Go ahead and book it.", 150, "accepted"),
        ("Yes, $180 is fine. Please book it.", 180, "accepted"),
        ("Yes please", 79, "accepted"),
        ("No problem, go ahead", 79, "accepted"),
        ("No, I don't want to pay $180. Keep my original price.", 180, "rejected"),
        ("I don't want it", 79, "rejected"),
        ("Yes, $150 is fine", 180, None),
        ("How long does it take?", 79, None),
    ],
)
def test_customer_replies_are_read_conservatively(message, price, expected):
    assert classify_reply(message, price) == expected


# The conversation runner wires this up for every backend.
def test_the_guarded_agent_follows_the_workflow_end_to_end():
    logs = run_conversation(
        ["Book an oil change Friday at 10am.", "Yes, $150 is fine.", "Yes, $180 is fine. Book it."], "C001", "guarded", CHANGE
    )
    assert [[c["name"] for c in log.tool_calls] for log in logs] == [
        ["check_availability", "get_quote"], ["validate_quote"], ["validate_quote", "book_appointment"],
    ]
    assert logs[2].tool_calls[-1]["result"]["status"] == "confirmed"


def test_the_naive_agent_is_still_stopped_by_the_tools():
    logs = run_conversation(
        ["Book an oil change Friday at 10am.", "Yes, $150 is fine."], "C001", "naive", CHANGE
    )
    assert not any(c["result"].get("status") == "confirmed" for log in logs for c in log.tool_calls)
