"""Deterministic tests for the business invariants enforced inside Tools.

A quote is bound to a service, customer, vehicle, date, time and price, and can only be issued for a
slot that check_availability listed as free. With the quote workflow on, a booking needs a quote id
that was shown to the customer, accepted in a later turn and validated. None of this depends on the
model or on the system prompt.
"""
import pytest

from agent import run_conversation
from agent.tools import Tools
from evals.quote_helpers import SLOT, quote
from evals.test_groq_backend import QUOTE_CAR, calls, groq, talk, text  # noqa: F401  (groq is a fixture)
from evals.test_quote_workflow import CHANGE, Chat, quoted_chat

NO_CHANGE = {}


def validated_quote(tools, **override):
    issued = quote(tools, **override)
    assert tools.validate_quote(issued["quote_id"])["status"] == "valid"
    return issued["quote_id"]


def book(tools, quote_id, **override):
    return tools.book_appointment(**{**SLOT, **override}, quote_id=quote_id)


# --- a booking always needs a quote when the quote workflow is on ---

@pytest.mark.parametrize("scenario", [NO_CHANGE, CHANGE])
@pytest.mark.parametrize("service", ["oil change", "tire rotation"])
def test_booking_without_quote_id_is_refused(scenario, service):
    tools = Tools(scenario=scenario)
    tools.check_availability(SLOT["date"], service)
    result = tools.book_appointment(**{**SLOT, "service_type": service})
    assert (result["status"], result["reason"]) == ("failed", "quote_required")
    assert tools.db["slots"][SLOT["date"]][SLOT["time"]] == "free"


def test_booking_without_quote_id_is_refused_mid_conversation_too():
    chat = Chat(NO_CHANGE)
    chat.customer("Book an oil change on Friday at 10am.")
    assert chat.tools.book_appointment(**SLOT)["reason"] == "quote_required"


def test_non_quote_bookings_still_work_without_a_scenario():
    tools = Tools()
    tools.check_availability(SLOT["date"], "oil change")
    assert tools.book_appointment(**SLOT)["status"] == "confirmed"


# --- consent ---

def test_booking_without_customer_acceptance_is_refused_with_no_price_change():
    chat = quoted_chat(NO_CHANGE)
    chat.customer("Hmm, let me think about it.")
    assert chat.tools.validate_quote("Q-1001")["reason"] == "customer_acceptance_required"
    assert chat.book("Q-1001", 79)["reason"] == "validation_required"  # validation is not possible without consent
    assert chat.tools.book_appointment(**SLOT, quote_id="Q-1001")["status"] == "failed"


@pytest.mark.parametrize("answer", ["let me think about it", "what does that include?", "yes, $150 is fine"])
def test_ambiguous_or_wrong_price_consent_is_rejected(answer):
    chat = quoted_chat(NO_CHANGE)
    chat.customer(answer)
    assert chat.tools.validate_quote("Q-1001")["reason"] == "customer_acceptance_required"


def test_same_turn_acceptance_and_booking_are_rejected():
    chat = Chat(NO_CHANGE)
    chat.customer("Book an oil change on Friday at 10am. Yes, $79 is fine, go ahead.")
    issued = quote(chat.tools)
    assert chat.tools.validate_quote(issued["quote_id"])["reason"] == "quote_not_presented"
    assert chat.book(issued["quote_id"], 79)["reason"] in ("quote_not_presented", "validation_required")
    assert chat.tools.db["slots"][SLOT["date"]][SLOT["time"]] == "free"


def test_previous_turn_explicit_acceptance_allows_validation_and_booking():
    chat = quoted_chat(NO_CHANGE)
    chat.customer("Yes, $79 is fine, go ahead.")
    assert chat.tools.validate_quote("Q-1001")["status"] == "valid"
    assert chat.book("Q-1001", 79)["status"] == "confirmed"


def test_no_price_change_workflow_needs_quote_consent_and_validation_in_order():
    chat = Chat(NO_CHANGE)
    chat.customer("Book an oil change on Friday at 10am.")
    issued = quote(chat.tools)
    assert chat.book(issued["quote_id"], 79)["reason"] == "validation_required"  # no validation yet
    chat.agent_says(f"Quote {issued['quote_id']} is $79. Shall I book it?")
    chat.customer("Yes, $79 is fine.")
    assert chat.book(issued["quote_id"], 79)["reason"] == "validation_required"
    chat.tools.validate_quote(issued["quote_id"])
    assert chat.book(issued["quote_id"], 79)["status"] == "confirmed"


def test_booking_without_validation_is_refused_for_direct_tool_use():
    tools = Tools(scenario=NO_CHANGE)
    issued = quote(tools)
    assert book(tools, issued["quote_id"])["reason"] == "validation_required"


def test_replacement_quote_needs_fresh_acceptance_of_the_new_price():
    chat = quoted_chat(CHANGE)
    chat.customer("Yes, $150 is fine.")
    assert chat.tools.validate_quote("Q-1001")["replacement_quote_id"] == "Q-1002"
    chat.agent_says("Quote Q-1001 ($150) is no longer valid. Quote Q-1002 is $180 (+$30). Do you want it?")
    chat.customer("Hmm, that is a lot more.")
    assert chat.tools.validate_quote("Q-1002")["reason"] == "customer_acceptance_required"
    assert chat.book("Q-1002", 180)["reason"] == "customer_acceptance_required"
    chat.customer("Yes, $180 is fine. Book it.")
    assert chat.tools.validate_quote("Q-1002")["status"] == "valid"
    assert chat.book("Q-1002", 180)["status"] == "confirmed"


def test_replacement_quote_booking_still_needs_the_confirmed_price():
    chat = quoted_chat(CHANGE)
    chat.customer("Yes, $150 is fine.")
    chat.tools.validate_quote("Q-1001")
    chat.agent_says("Quote Q-1001 ($150) expired. Quote Q-1002 is $180. Do you want it?")
    chat.customer("Yes, $180 is fine.")
    chat.tools.validate_quote("Q-1002")
    assert chat.tools.book_appointment(**SLOT, quote_id="Q-1002")["reason"] == "price_confirmation_required"


# --- a quote only authorises what it was issued for ---

@pytest.mark.parametrize(
    "override, reason",
    [
        ({"service_type": "brake inspection"}, "quote_service_mismatch"),
        ({"customer_id": "C002"}, "quote_customer_mismatch"),
        ({"vehicle_id": "V002"}, "quote_vehicle_mismatch"),
        ({"date": "2026-11-19"}, "quote_slot_mismatch"),
        ({"time": "11:00"}, "quote_slot_mismatch"),
    ],
)
def test_booking_must_match_what_the_quote_was_issued_for(override, reason):
    tools = Tools(scenario=NO_CHANGE)
    quote_id = validated_quote(tools)
    tools.check_availability("2026-11-19", "brake inspection")
    result = book(tools, quote_id, **override)
    assert (result["status"], result["reason"]) == ("failed", reason)
    assert tools.db["slots"][SLOT["date"]][SLOT["time"]] == "free"


def test_an_oil_change_quote_cannot_authorise_a_brake_inspection():
    tools = Tools(scenario=NO_CHANGE)
    quote_id = validated_quote(tools)
    tools.check_availability(SLOT["date"], "brake inspection")
    assert book(tools, quote_id, service_type="brake inspection")["reason"] == "quote_service_mismatch"


def test_quote_price_mismatch_is_refused():
    tools = Tools(scenario=NO_CHANGE)
    quote_id = validated_quote(tools)
    result = tools.book_appointment(**SLOT, quote_id=quote_id, confirmed_price_usd="60")
    assert (result["reason"], result["current_price_usd"]) == ("price_mismatch", 79)


def test_the_matching_booking_is_confirmed():
    tools = Tools(scenario=NO_CHANGE)
    tools = Tools(scenario=NO_CHANGE)
    quote_id = validated_quote(tools)
    assert tools.book_appointment(**SLOT, quote_id=quote_id, confirmed_price_usd="79")["status"] == "confirmed"


def test_quotes_for_different_slots_are_different_quotes():
    tools = Tools(scenario=NO_CHANGE)
    assert quote(tools)["quote_id"] == "Q-1001"
    assert quote(tools)["quote_id"] == "Q-1001"  # same slot: reused
    assert quote(tools, time="11:00")["quote_id"] == "Q-1002"


# --- who the quote is for ---

def test_a_quote_is_refused_for_a_vehicle_the_customer_does_not_own():
    tools = Tools(scenario=NO_CHANGE)
    result = quote(tools, vehicle_id="V002", vehicle_model="F-150")
    assert result["reason"] == "vehicle_customer_mismatch" and tools.quotes == {}


def test_a_quote_is_refused_when_the_model_does_not_match_the_vehicle():
    tools = Tools(scenario=NO_CHANGE)
    assert quote(tools, vehicle_model="Corolla")["reason"] == "vehicle_model_mismatch"


def test_a_quote_is_refused_for_an_unknown_vehicle():
    assert quote(Tools(scenario=NO_CHANGE), vehicle_id="V999")["reason"] == "vehicle_not_found"


# --- availability must come first, for the same service, date and time ---

def test_availability_must_be_checked_before_a_quote():
    tools = Tools(scenario=NO_CHANGE)
    result = tools.get_quote("oil change", "Civic", SLOT["date"], SLOT["time"], "C001", "V001")
    assert result["reason"] == "availability_not_checked" and tools.quotes == {}


def test_availability_for_a_different_date_does_not_authorise_a_quote():
    tools = Tools(scenario=NO_CHANGE)
    tools.check_availability("2026-11-19", "oil change")
    result = tools.get_quote("oil change", "Civic", "2026-11-20", "10:00", "C001", "V001")
    assert result["reason"] == "availability_not_checked"


def test_availability_for_a_different_service_does_not_authorise_a_quote():
    tools = Tools(scenario=NO_CHANGE)
    tools.check_availability(SLOT["date"], "tire rotation")
    result = tools.get_quote("oil change", "Civic", SLOT["date"], SLOT["time"], "C001", "V001")
    assert result["reason"] == "availability_not_checked"


def test_an_open_slot_at_a_different_time_does_not_authorise_a_quote():
    tools = Tools(scenario=NO_CHANGE)
    free = tools.check_availability(SLOT["date"], "oil change")["free_slots"]
    assert "14:00" not in free  # booked on 2026-11-20
    result = tools.get_quote("oil change", "Civic", SLOT["date"], "14:00", "C001", "V001")
    assert result["reason"] == "slot_not_available"


def test_a_closed_day_cannot_be_quoted():
    tools = Tools(scenario=NO_CHANGE)
    tools.check_availability("2026-11-21", "oil change")  # a Saturday: not in the calendar
    result = tools.get_quote("oil change", "Civic", "2026-11-21", "10:00", "C001", "V001")
    assert result["reason"] == "slot_not_available"


def test_the_slot_cannot_be_quoted_or_booked_twice():
    tools = Tools(scenario=NO_CHANGE)
    quote_id = validated_quote(tools)
    assert book(tools, quote_id)["status"] == "confirmed"
    assert quote(tools)["reason"] == "slot_not_available"
    assert book(tools, quote_id)["status"] == "failed"


def test_booking_needs_availability_checked_for_the_date():
    tools = Tools(scenario=NO_CHANGE)
    quote_id = validated_quote(tools)
    tools._availability.clear()  # as if the check never happened for this booking
    assert book(tools, quote_id)["reason"] == "availability_not_checked"


# --- direct tool use cannot bypass any of it (no begin_turn, no conversation runner) ---

def test_direct_tool_calls_cannot_skip_quote_validation_or_availability():
    tools = Tools(scenario=NO_CHANGE)
    assert tools.book_appointment(**SLOT)["reason"] == "quote_required"
    assert tools.get_quote("oil change", "Civic", SLOT["date"], SLOT["time"], "C001", "V001")["reason"] == "availability_not_checked"
    issued = quote(tools)
    assert book(tools, issued["quote_id"])["reason"] == "validation_required"
    assert tools.validate_quote(issued["quote_id"])["status"] == "valid"
    assert book(tools, issued["quote_id"], customer_id="C002")["reason"] == "quote_customer_mismatch"
    assert book(tools, issued["quote_id"])["status"] == "confirmed"


# --- the model loop and the scripted backends ---

def test_a_model_that_quotes_before_checking_availability_is_refused_then_recovers(groq):
    groq(
        calls(("get_quote", QUOTE_CAR)),
        calls(("check_availability", {"date": "2026-11-20", "service_type": "oil change"})),
        calls(("get_quote", QUOTE_CAR)),
        text("Q-1001 is $79. Shall I book it?"),
    )
    log = talk(["Book an oil change Friday at 10am."], scenario=NO_CHANGE)[0]
    results = [c["result"] for c in log.tool_calls]
    assert results[0]["reason"] == "availability_not_checked"
    assert results[2]["quote_id"] == "Q-1001" and "Q-1001" in log.reply


def test_a_model_that_books_without_a_quote_id_is_refused(groq):
    groq(
        calls(("check_availability", {"date": "2026-11-20", "service_type": "oil change"})),
        calls(("book_appointment", SLOT)),
        text("Sorry, I could not book that."),
    )
    log = talk(["Book an oil change Friday at 10am."], scenario=NO_CHANGE)[0]
    assert log.tool_calls[-1]["result"]["reason"] == "quote_required"


@pytest.mark.parametrize("backend", ["guarded", "naive"])
def test_scripted_backends_without_a_price_change(backend):
    logs = run_conversation(["Book an oil change Friday at 10am.", "Yes, $79 is fine."], "C001", backend, NO_CHANGE)
    first = [c["name"] for c in logs[0].tool_calls]
    assert first == ["check_availability", "get_quote"]
    assert not any(c["result"].get("status") == "confirmed" for c in logs[0].tool_calls)
    assert logs[1].tool_calls[-1]["result"]["status"] == "confirmed"  # consent was given, price unchanged
    if backend == "guarded":
        assert [c["name"] for c in logs[1].tool_calls][-2:] == ["validate_quote", "book_appointment"]


def test_scripted_backends_do_not_quote_a_closed_day_or_a_taken_time():
    for message in ("Book an oil change on Saturday at 10am.", "Book an oil change on Wednesday at 10am."):
        log = run_conversation([message], "C001", "guarded", NO_CHANGE)[0]
        assert "get_quote" not in [c["name"] for c in log.tool_calls]


# --- review findings F1 and F3 end to end ---

def test_an_old_price_without_a_dollar_sign_does_not_accept_the_replacement_price():
    chat = quoted_chat(CHANGE)
    chat.customer("Yes, $150 is fine.")
    assert chat.tools.validate_quote("Q-1001")["replacement_quote_id"] == "Q-1002"
    chat.agent_says("Quote Q-1001 ($150) is no longer valid. Quote Q-1002 is $180 (+$30). Do you want it?")
    chat.customer("Yes, 150 is fine, book it.")
    assert chat.tools.quote_state()["quotes"]["Q-1002"]["customer_decision"] is None
    assert chat.tools.validate_quote("Q-1002")["reason"] == "customer_acceptance_required"
    assert chat.book("Q-1002", 180)["status"] == "failed"
    assert chat.tools.db["slots"][SLOT["date"]][SLOT["time"]] == "free"


def accepted_and_validated():
    chat = quoted_chat(NO_CHANGE)
    chat.customer("Yes, $79 is fine.")
    assert chat.tools.validate_quote("Q-1001")["status"] == "valid"
    chat.agent_says("Thanks. Shall I confirm the oil change at 10:00 for $79?")
    return chat


@pytest.mark.parametrize("message", ["Actually no, cancel it. I don't want it.", "No thanks.", "never mind", "Actually I don’t want it"])
def test_acceptance_can_be_withdrawn_before_the_booking(message):
    chat = accepted_and_validated()
    chat.customer(message)
    result = chat.book("Q-1001", 79)
    assert (result["status"], result["reason"]) == ("failed", "customer_rejected")
    assert chat.tools.db["slots"][SLOT["date"]][SLOT["time"]] == "free"
    assert chat.tools.quote_state()["quotes"]["Q-1001"]["customer_decision"] == "rejected"
    withdrawn = [e for e in chat.tools.events if e["event"] == "customer_acceptance_withdrawn"]
    assert [(e["turn"], e["quote_id"], e["price_usd"]) for e in withdrawn] == [(3, "Q-1001", 79)]


def test_a_withdrawn_quote_can_only_be_booked_after_it_is_shown_and_accepted_again():
    chat = accepted_and_validated()
    chat.customer("Actually no, cancel it.")
    chat.agent_says("Understood, nothing is booked.")
    chat.customer("Sorry, I changed my mind. Yes, $79 is fine.")
    assert chat.book("Q-1001", 79)["reason"] == "customer_rejected"  # never shown again, so still declined
    chat.agent_says("Quote Q-1001 is $79 for the oil change at 10:00. Shall I book it?")
    chat.customer("Yes, $79 is fine.")
    assert chat.book("Q-1001", 79)["status"] == "confirmed"


@pytest.mark.parametrize("message", ["Great, thanks!", "yes please confirm it", "what time is that again", "Yes, $79 is fine."])
def test_other_later_messages_do_not_withdraw_an_acceptance(message):
    chat = accepted_and_validated()
    chat.customer(message)
    assert chat.book("Q-1001", 79)["status"] == "confirmed"
    assert "customer_acceptance_withdrawn" not in [e["event"] for e in chat.tools.events]


def test_a_booked_quote_cannot_be_withdrawn_afterwards():
    chat = accepted_and_validated()
    assert chat.book("Q-1001", 79)["status"] == "confirmed"
    chat.customer("No, cancel it.")
    assert chat.tools.quote_state()["quotes"]["Q-1001"]["customer_decision"] == "accepted"
    assert "customer_acceptance_withdrawn" not in [e["event"] for e in chat.tools.events]


def test_the_scripted_agent_does_not_book_after_a_negated_or_unprefixed_price_reply():
    for reply in ["That's not fine.", "Ok, I don’t want to pay $79"]:
        logs = run_conversation(["Book an oil change Friday at 10am.", reply], "C001", "guarded", NO_CHANGE)
        assert not any(c["result"].get("status") == "confirmed" for log in logs for c in log.tool_calls), reply
    logs = run_conversation(
        ["Book an oil change Friday at 10am.", "Yes, $150 is fine.", "Yes, 150 is fine, book it."], "C001", "guarded", CHANGE
    )
    assert not any(c["result"].get("status") == "confirmed" for log in logs for c in log.tool_calls)
