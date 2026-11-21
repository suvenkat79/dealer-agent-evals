"""Deterministic tests for quote sequencing, id normalisation and the evaluator's new checks."""
from types import SimpleNamespace

from agent import run_conversation
from agent.tools import Tools, normalize_id
from evals.quote_helpers import quote
from evals.runner import check_log

CHANGE = {"price_change": {"service_type": "oil change", "original_usd": 150, "updated_usd": 180}}


def quoted():
    tools = Tools(scenario=CHANGE)
    quote(tools)
    return tools


def test_get_quote_is_refused_until_the_expired_original_is_validated():
    tools = quoted()
    result = quote(tools)
    assert "validate_quote" in result["error"] and "Q-1001" in result["error"]
    assert set(tools.quotes) == {"Q-1001"}


def test_original_quote_state_is_preserved_and_reported():
    tools = quoted()
    check = tools.validate_quote("Q-1001")
    assert (check["original_quote_id"], check["original_price_usd"]) == ("Q-1001", 150)
    assert tools.validate_quote("Q-1002")["original_quote_id"] == "Q-1001"
    assert tools.quote_state()["original_quote_id"] == "Q-1001"
    assert quote(tools)["quote_id"] == "Q-1002"
    assert tools.original_quote_id == "Q-1001"


def test_repeat_validation_is_flagged_and_issues_no_new_quote():
    tools = quoted()
    first = tools.validate_quote("Q-1001")
    again = tools.validate_quote("Q-1001")
    assert "already_validated" not in first and again["already_validated"] is True
    assert again["replacement_quote_id"] == "Q-1002" and set(tools.quotes) == {"Q-1001", "Q-1002"}


def test_look_alike_hyphens_are_normalised_but_different_ids_are_not():
    assert normalize_id("Q‑1001") == normalize_id("Q–001".replace("001", "1001")) == "Q-1001"
    assert normalize_id("Q-1003") == "Q-1003"
    tools = quoted()
    assert tools.validate_quote("Q‑1001")["status"] == "expired"
    assert "not found" in tools.validate_quote("Q-1003")["error"]


def _turn(reply, calls=()):
    return SimpleNamespace(reply=reply, tool_calls=list(calls))


def _call(name, result, **args):
    return {"name": name, "args": args, "result": result}


def test_evaluator_accepts_a_unicode_hyphen_in_a_real_quote_id():
    calls = [_call("get_quote", {"quote_id": "Q-1001", "price_usd": 150})]
    assert check_log(_turn("Your quote is Q‑1001 at $150.", calls), {}) == []


def test_evaluator_rejects_an_id_no_tool_issued():
    calls = [_call("get_quote", {"quote_id": "Q-1002", "price_usd": 180})]
    failures = check_log(_turn("Booked under Q-1003 at $180.", calls), {})
    assert any("Q-1003" in f for f in failures)


def test_evaluator_flags_duplicate_validation_in_one_turn():
    result = {"quote_id": "Q-1001"}
    calls = [_call("validate_quote", result, quote_id="Q-1001"), _call("validate_quote", result, quote_id="Q‑1001")]
    assert any("more than once" in f for f in check_log(_turn("ok", calls), {}))


def test_prices_may_be_grounded_in_earlier_turns():
    prior = [_call("get_quote", {"quote_id": "Q-1001", "price_usd": 150})]
    assert check_log(_turn("The original price was $150."), {}, prior_calls=prior) == []
    assert check_log(_turn("The original price was $150."), {}) != []


def test_guarded_agent_never_requests_a_second_quote_after_a_price_change():
    logs = run_conversation(
        ["I want to book an oil change on Friday at 10am.", "Yes, $150 is fine.", "Yes, $180 is fine. Book it."],
        "C001", "guarded", CHANGE,
    )
    sequence = [[c["name"] for c in log.tool_calls] for log in logs]
    assert sequence == [["check_availability", "get_quote"], ["validate_quote"], ["validate_quote", "book_appointment"]]
    assert "Q-1001" in logs[1].reply and "Q-1002" in logs[1].reply and "APT-0001" in logs[2].reply


def test_evaluator_flags_booking_before_validating_the_quote():
    calls = [_call("check_availability", {}), _call("book_appointment", {"status": "failed"}, quote_id="Q-1001")]
    assert any("without validate_quote" in f for f in check_log(_turn("ok", calls), {}))
