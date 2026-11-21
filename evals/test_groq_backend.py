"""Deterministic tests of the Groq backend with a scripted fake client. No network, no API key.

The real run_groq loop, retry logic and tools run; only the model's answers are scripted. Tests
against the live API are in test_agent_evals.py (AGENT_BACKEND=groq).
"""
import copy
import itertools
import json
from types import SimpleNamespace

import pytest

from agent import llm_backend, run_conversation
from agent.errors import BackendError
from evals.runner import run_case

CHANGE = {"price_change": {"service_type": "oil change", "original_usd": 150, "updated_usd": 180}}
SLOT = {"customer_id": "C001", "vehicle_id": "V001", "date": "2026-11-20", "time": "10:00", "service_type": "oil change"}
CAR = {"service_type": "oil change", "vehicle_model": "Civic"}
QUOTE_CAR = {**CAR, "date": "2026-11-20", "time": "10:00", "customer_id": "C001", "vehicle_id": "V001"}
_ids = itertools.count(1)


def text(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None))])


def calls(*requests):
    tool_calls = [
        SimpleNamespace(id=f"call_{next(_ids)}", function=SimpleNamespace(name=name, arguments=json.dumps(args)))
        for name, args in requests
    ]
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=tool_calls))])


class FakeClient:
    """Plays back scripted model responses (or raises scripted exceptions), recording each request."""

    def __init__(self, *steps):
        self.steps = list(steps)
        self.requests = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs["messages"]))
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


class RateLimited(Exception):
    def __init__(self, retry_after=None):
        super().__init__("429 rate limit")
        self.status_code = 429
        self.response = SimpleNamespace(headers={} if retry_after is None else {"retry-after": str(retry_after)})


@pytest.fixture
def groq(monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm_backend, "_sleep", sleeps.append)

    def install(*steps):
        client = FakeClient(*steps)
        monkeypatch.setattr(llm_backend, "_make_client", lambda: client)
        return client

    install.sleeps = sleeps
    return install


def talk(messages, scenario=CHANGE):
    return run_conversation(messages, "C001", "groq", scenario)


def names(log):
    return [c["name"] for c in log.tool_calls]


def confirmed(logs):
    return [c for log in logs for c in log.tool_calls if c["result"].get("status") == "confirmed"]


QUOTE_TURN = [calls(("check_availability", {"date": "2026-11-20", "service_type": "oil change"}), ("get_quote", QUOTE_CAR)), text("Q-1001 is $150.")]
CHANGED_TURN = [calls(("validate_quote", {"quote_id": "Q-1001"})), text("Q-1001 ($150) expired; now $180 (+$30) as Q-1002. Accept?")]


# --- business rules, end to end through the LLM loop ---

def test_accepted_replacement_books_on_the_replacement_quote_and_the_reply_is_not_edited(groq):
    groq(
        *QUOTE_TURN, *CHANGED_TURN,
        calls(("validate_quote", {"quote_id": "Q-1002"})),
        calls(("book_appointment", {**SLOT, "quote_id": "Q-1002", "confirmed_price_usd": "180"})),
        text("All set, you are booked."),  # leaves out the id and the price
    )
    logs = talk(["book oil change friday 10am", "yes $150", "yes $180"])
    booking = confirmed(logs)[0]
    assert booking["args"]["quote_id"] == "Q-1002" and booking["result"]["price_usd"] == 180
    assert logs[2].reply == "All set, you are booked."  # no appended confirmation: a vague reply stays vague
    assert "get_quote" not in names(logs[2])


def test_booking_in_the_same_response_as_validation_is_not_executed(groq):
    groq(
        *QUOTE_TURN,
        calls(("validate_quote", {"quote_id": "Q-1001"}), ("book_appointment", {**SLOT, "quote_id": "Q-1001", "confirmed_price_usd": "150"})),
        text("The price changed, nothing is booked."),
    )
    logs = talk(["book", "yes $150"])
    booking = [c for c in logs[1].tool_calls if c["name"] == "book_appointment"][0]
    assert booking["result"]["reason"] == "not_executed"
    assert not confirmed(logs)


def test_independent_read_only_calls_in_one_response_all_run(groq):
    groq(
        calls(("check_availability", {"date": "2026-11-20", "service_type": "oil change"}), ("estimate_cost", CAR), ("get_vehicle_history", {"vehicle_id": "V001"})),
        text("ok"),
    )
    log = talk(["what is free"], scenario=None)[0]
    assert all("error" not in c["result"] for c in log.tool_calls) and len(log.tool_calls) == 3


def test_replacement_cannot_be_booked_in_the_turn_it_was_issued(groq):
    # A model that assumes the customer will accept tries to book the replacement straight away.
    groq(
        *QUOTE_TURN,
        calls(("validate_quote", {"quote_id": "Q-1001"})),
        calls(("book_appointment", {**SLOT, "quote_id": "Q-1002", "confirmed_price_usd": "180"})),
        text("I could not book yet."),
    )
    logs = talk(["book", "yes $150"])
    assert logs[1].tool_calls[-1]["result"]["reason"] == "quote_not_presented"
    assert not confirmed(logs)


def test_rejected_price_leads_to_no_booking(groq):
    groq(*QUOTE_TURN, *CHANGED_TURN, text("Understood. Quote Q-1001 is no longer valid and no appointment has been booked."))
    logs = talk(["book", "yes $150", "No, I don't want to pay $180."])
    assert names(logs[2]) == [] and not confirmed(logs)


def test_unchanged_price_books_on_the_original_quote(groq):
    groq(
        calls(("check_availability", {"date": "2026-11-20", "service_type": "oil change"}), ("get_quote", QUOTE_CAR)),
        text("Q-1001 is $79."),
        calls(("validate_quote", {"quote_id": "Q-1001"})),
        calls(("book_appointment", {**SLOT, "quote_id": "Q-1001", "confirmed_price_usd": "79"})),
        text("Booked."),
    )
    logs = talk(["book", "yes"], scenario={})
    booking = confirmed(logs)[0]
    assert booking["args"]["quote_id"] == "Q-1001" and booking["result"]["price_usd"] == 79


def test_unknown_quote_cannot_be_booked(groq):
    groq(
        calls(("check_availability", {"date": "2026-11-20", "service_type": "oil change"})),
        calls(("validate_quote", {"quote_id": "Q-9999"})),
        calls(("book_appointment", {**SLOT, "quote_id": "Q-9999", "confirmed_price_usd": "150"})),
        text("Sorry."),
    )
    log = talk(["book"])[0]
    assert "not found" in log.tool_calls[1]["result"]["error"]
    assert log.tool_calls[2]["result"]["reason"] == "unknown_quote"


def test_original_quote_must_be_validated_before_booking(groq):
    groq(
        *QUOTE_TURN,
        calls(("book_appointment", {**SLOT, "quote_id": "Q-1001", "confirmed_price_usd": "150"})),
        text("Sorry."),
    )
    logs = talk(["book", "yes"])
    assert logs[1].tool_calls[0]["result"]["reason"] == "validation_required"


# --- conversation history ---

def assert_valid_tool_history(messages):
    """Every assistant tool_calls message is followed by exactly one tool message per call id."""
    i = 1 if messages[0]["role"] == "system" else 0
    while i < len(messages):
        message = messages[i]
        if message.get("tool_calls"):
            wanted = [c["id"] for c in message["tool_calls"]]
            got = [m["tool_call_id"] for m in messages[i + 1 : i + 1 + len(wanted)] if m["role"] == "tool"]
            assert got == wanted, f"tool results {got} do not match calls {wanted}"
            i += len(wanted)
        else:
            assert message["role"] != "tool", "orphan tool message"
        i += 1


def test_tool_calls_and_results_carry_over_to_later_turns(groq):
    client = groq(*QUOTE_TURN, *CHANGED_TURN, text("Understood, nothing booked."))
    talk(["book", "yes $150", "no"])
    last = client.requests[-1]
    assert_valid_tool_history(last)
    tool_texts = [m["content"] for m in last if m["role"] == "tool"]
    assert any("Q-1001" in t and "replacement_quote_id" in t for t in tool_texts)  # turn 2's validation result
    assert [m["content"] for m in last if m["role"] == "user"] == ["book", "yes $150", "no"]  # no duplicates
    for request in client.requests:
        assert_valid_tool_history(request)


def test_deferred_booking_still_gets_a_tool_message(groq):
    client = groq(
        *QUOTE_TURN,
        calls(("validate_quote", {"quote_id": "Q-1001"}), ("book_appointment", {**SLOT, "quote_id": "Q-1001"})),
        text("ok"),
        text("fine"),
    )
    talk(["book", "yes"])
    assert_valid_tool_history(client.requests[-1])


def test_history_is_untouched_when_a_turn_fails(groq):
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    groq(RateLimited(retry_after=9999))
    from agent.agent import LoggedTools, RunLog
    from agent.tools import Tools

    with pytest.raises(BackendError):
        llm_backend.run_groq("book", "C001", LoggedTools(Tools(scenario={}), RunLog("book", "C001", "groq")), history=history, quoting=True)
    assert len(history) == 2


# --- quote workflow detection ---

def test_quote_rules_only_apply_when_the_conversation_has_a_quote_workflow(groq):
    client = groq(text("hi"), text("hi"))
    talk(["hello"], scenario=None)
    talk(["hello"], scenario={})
    assert "validate_quote" not in client.requests[0][0]["content"]
    assert "validate_quote" in client.requests[1][0]["content"]
    assert "Quote state from the booking system" in client.requests[1][0]["content"]


# --- rate limits ---

def test_a_429_is_retried_after_the_advised_delay(groq):
    client = groq(RateLimited(retry_after=2), text("hello"))
    assert talk(["hi"], scenario=None)[0].reply == "hello"
    assert len(client.requests) == 2 and len(groq.sleeps) == 1 and 2 <= groq.sleeps[0] <= 3


def test_backoff_grows_and_retries_are_bounded(groq):
    client = groq(*[RateLimited() for _ in range(10)])
    with pytest.raises(BackendError) as info:
        talk(["hi"], scenario=None)
    assert info.value.kind == "rate_limit"
    assert len(client.requests) == 4 and len(groq.sleeps) == 3  # 1 try + GROQ_MAX_RETRIES (3)
    assert 1 <= groq.sleeps[0] <= 2 and 2 <= groq.sleeps[1] <= 4 and 4 <= groq.sleeps[2] <= 8


def test_a_wait_longer_than_the_cap_fails_immediately(groq):
    client = groq(RateLimited(retry_after=900), text("never reached"))
    with pytest.raises(BackendError, match="retry-after=900"):
        talk(["hi"], scenario=None)
    assert len(client.requests) == 1 and not groq.sleeps


def test_retry_settings_come_from_the_environment(groq, monkeypatch):
    monkeypatch.setenv("GROQ_MAX_RETRIES", "1")
    client = groq(RateLimited(), RateLimited(), text("unused"))
    with pytest.raises(BackendError):
        talk(["hi"], scenario=None)
    assert len(client.requests) == 2


def test_other_api_errors_are_not_retried(groq):
    client = groq(ConnectionError("boom"), text("unused"))
    with pytest.raises(ConnectionError):
        talk(["hi"], scenario=None)
    assert len(client.requests) == 1


def test_the_real_groq_rate_limit_exception_is_recognised(groq):
    pytest.importorskip("groq")
    import groq as sdk
    import httpx

    response = httpx.Response(429, headers={"retry-after": "1"}, request=httpx.Request("POST", "https://api.groq.com"))
    client = groq(sdk.RateLimitError("limit", response=response, body=None), text("ok"))
    assert talk(["hi"], scenario=None)[0].reply == "ok" and len(client.requests) == 2


def test_a_rate_limited_case_is_a_tagged_failure_never_a_pass(groq):
    groq(*[RateLimited() for _ in range(4)])
    case = {"id": "x", "customer_id": "C001", "message": "hi", "expect": {}}
    log, failures = run_case(case, "groq")
    assert len(failures) == 1 and failures[0].startswith("[external:rate_limit]") and log.reply == ""


def test_step_limit_is_configurable(groq, monkeypatch):
    monkeypatch.setenv("GROQ_MAX_STEPS", "2")
    client = groq(*[calls(("estimate_cost", CAR)) for _ in range(5)])
    talk(["cost"], scenario=None)
    assert len(client.requests) == 2


# --- the scripted backends keep their behaviour ---

@pytest.mark.parametrize(
    "backend, expected",
    [
        ("guarded", [["check_availability", "get_quote"], ["validate_quote"], ["validate_quote", "book_appointment"]]),
        ("naive", [["check_availability", "get_quote"], ["book_appointment", "validate_quote", "book_appointment"]]),
    ],
)
def test_scripted_backends_keep_their_tool_sequences(backend, expected):
    messages = ["I want to book an oil change on Friday at 10am.", "Yes, $150 is fine.", "Yes, $180 is fine. Book it."]
    logs = run_conversation(messages[: len(expected)], "C001", backend, CHANGE)
    assert [names(log) for log in logs] == expected


def test_missing_api_key_is_a_tagged_failure(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    log, failures = run_case({"id": "x", "customer_id": "C001", "message": "hi", "expect": {}}, "groq")
    assert failures[0].startswith("[external:missing_api_key]")
