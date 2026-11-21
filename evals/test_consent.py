"""Consent is read conservatively: only a clear, unhedged yes at the quoted price counts."""
import pytest

from agent import run_conversation
from agent.consent import classify_reply

ACCEPTED = [
    "Yes, $79 is fine. Go ahead and book it.",
    "Yes, $180 is fine. Please book it.",
    "yes", "ok", "okay", "sure", "fine", "go ahead", "yes please", "book it", "That works.",
]
AMBIGUOUS = [
    "ok maybe",
    "okay, I guess",
    "sure, maybe later",
    "I'm not sure, but ok",
    "ok but is there a cheaper option?",
    "yes?",
    "Can you book it?",
    "fine, I suppose",
    "yeah, I think so",
    "It depends, go ahead if it is quick",
    "hold on, ok",
    "ok however I wanted morning",
    "let me think about it",
    "what does that include?",
]
REJECTED = ["No", "No, maybe later", "no thanks, too expensive", "I don't want that", "never mind, cancel"]


@pytest.mark.parametrize("message", ACCEPTED)
def test_clear_acceptance_is_accepted(message):
    assert classify_reply(message, 79 if "180" not in message else 180) == "accepted"


@pytest.mark.parametrize("message", AMBIGUOUS)
def test_hedged_or_conditional_replies_are_not_acceptance(message):
    assert classify_reply(message, 79) is None


@pytest.mark.parametrize("message", REJECTED)
def test_refusals_stay_refusals_even_when_hedged(message):
    assert classify_reply(message, 79) == "rejected"


def test_a_yes_to_a_different_price_is_not_acceptance():
    assert classify_reply("Yes, $150 is fine.", 180) is None


@pytest.mark.parametrize("message", ["ok maybe", "okay, I guess", "I'm not sure, but ok"])
def test_the_scripted_agent_does_not_book_on_a_hedged_yes(message):
    logs = run_conversation(["Book an oil change Friday at 10am.", message], "C001", "guarded", {})
    assert not any(c["result"].get("status") == "confirmed" for log in logs for c in log.tool_calls)
    assert "reply yes" in logs[1].reply


def test_the_scripted_agent_still_books_on_a_clear_yes():
    logs = run_conversation(["Book an oil change Friday at 10am.", "Yes, $79 is fine."], "C001", "guarded", {})
    assert logs[1].tool_calls[-1]["result"]["status"] == "confirmed"


# --- review findings F1 and F2: a price without "$", negations, conditions, curly apostrophes ---

@pytest.mark.parametrize(
    "message",
    ["Yes, 150 is fine", "Yes, 150 dollars is fine", "ok, I'll pay 150 USD", "Yes, one fifty is fine", "yes $150.00 is fine", "Yes, 150 is fine, book it."],
)
def test_an_old_price_in_any_form_is_not_acceptance_of_the_new_price(message):
    assert classify_reply(message, 180) is None


@pytest.mark.parametrize("message", ["Yes, 79 is fine", "yes $79.00 is fine", "Yes, $79 is fine. Go ahead and book it."])
def test_the_quoted_price_is_accepted_with_or_without_the_dollar_sign(message):
    assert classify_reply(message, 79) == "accepted"


@pytest.mark.parametrize(
    "message",
    ["Yes, $79 is fine. Book it for 10am on Friday.", "yes, 79 is fine, the 20th at 10:00", "Yes Q-1001 at $79 is fine", "Yes, $1,200 is fine"],
)
def test_times_dates_and_ids_are_not_mistaken_for_prices(message):
    price = 1200 if "1,200" in message else 79
    assert classify_reply(message, price) == "accepted"


@pytest.mark.parametrize("message", ["$150 or $180, either is fine", "Yes, $150 is fine and $180 is fine"])
def test_every_amount_mentioned_must_be_the_quoted_price(message):
    assert classify_reply(message, 180) is None


@pytest.mark.parametrize(
    "message",
    [
        "That's not fine.",
        "I'm not okay with that",
        "not ok",
        "that is not fine at all",
        "Ok, I don’t want to pay $79",
        "Sure, if it takes under an hour",
        "Sure, as long as it's done by noon",
        "ok unless it rains",
        "probably yes",
        "fine, whatever",
        "it doesn't work for me, ok",
        "never ok",
    ],
)
def test_negated_conditional_or_curly_quote_replies_are_never_acceptance(message):
    assert classify_reply(message, 79) != "accepted"


@pytest.mark.parametrize("message", ["Ok, I don’t want to pay $79", "I don’t want it, but ok", "DON’T book it, ok?"])
def test_a_phone_keyboards_curly_apostrophe_still_declines(message):
    assert classify_reply(message, 79) in ("rejected", None)


def test_the_curly_apostrophe_refusal_is_a_rejection_not_just_unclear():
    assert classify_reply("Ok, I don’t want to pay $79", 79) == "rejected"


@pytest.mark.parametrize("message", ["no problem, go ahead", "No worries, book it", "Yes, no problem"])
def test_no_problem_is_still_a_yes(message):
    assert classify_reply(message, 79) == "accepted"
