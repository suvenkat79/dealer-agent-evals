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
