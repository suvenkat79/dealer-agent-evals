"""Token usage, retry and rate-limit telemetry, recorded per model call. Fake client only: no network."""
import json
from types import SimpleNamespace

import pytest

from agent import llm_backend, run_conversation
from agent.errors import BackendError
from agent.telemetry import CallRecord, summarize
from evals.runner import load_cases, run_case, write_telemetry
from evals.test_groq_backend import QUOTE_CAR, FakeClient, RateLimited, calls, groq, talk, text  # noqa: F401

CHECK = ("check_availability", {"date": "2026-11-20", "service_type": "oil change"})


def used(response, prompt, completion):
    response.usage = SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion)
    return response


class Limited(RateLimited):
    """A 429 with the headers Groq sends, plus one that must never be copied."""

    def __init__(self, retry_after=None):
        super().__init__(retry_after)
        self.response.headers.update(
            {"x-ratelimit-remaining-tokens": "12", "x-ratelimit-reset-tokens": "5.2s", "authorization": "Bearer gsk_SECRET"}
        )


def test_token_usage_is_recorded_per_model_call(groq):
    groq(used(calls(CHECK, ("get_quote", QUOTE_CAR)), 1553, 103), used(text("Q-1001 is $79."), 1627, 100))
    log = talk(["Book an oil change Friday at 10am."], scenario={})[0]
    first, second = log.llm_calls
    assert (first.turn, first.step, first.outcome) == (1, 1, "ok")
    assert (first.prompt_tokens, first.completion_tokens, first.total_tokens) == (1553, 103, 1656)
    assert (second.step, second.total_tokens) == (2, 1727)
    assert summarize(log.llm_calls)["total_tokens"] == 3383


def test_a_response_without_usage_is_recorded_as_unknown_not_zero(groq):
    groq(text("hello"))
    record = talk(["hi"], scenario=None)[0].llm_calls[0]
    assert (record.prompt_tokens, record.completion_tokens, record.total_tokens) == (None, None, None)
    totals = summarize([record])
    assert totals["total_tokens"] is None and totals["calls_without_usage"] == 1


def test_a_recovered_429_records_the_retry_wait_and_the_rate_limit_headers(groq):
    groq(Limited(retry_after=3), Limited(retry_after=2), used(text("hello"), 100, 5))
    record = talk(["hi"], scenario=None)[0].llm_calls[0]
    assert (record.outcome, record.attempts, record.retry_count) == ("ok", 3, 2)
    assert record.retry_wait_s == pytest.approx(sum(groq.sleeps), abs=0.01) and 5 <= record.retry_wait_s <= 7
    event = record.rate_limit_events[0]
    assert (event["attempt"], event["status"], event["retry_after_s"], event["action"]) == (1, 429, 3.0, "retry")
    assert event["headers"] == {"retry-after": "3", "x-ratelimit-remaining-tokens": "12", "x-ratelimit-reset-tokens": "5.2s"}
    assert record.total_tokens == 105


def test_a_wait_over_the_cap_is_recorded_as_given_up_without_sleeping(groq):
    groq(Limited(retry_after=682))
    with pytest.raises(BackendError) as caught:
        talk(["hi"], scenario=None)
    record = caught.value.logs[0].llm_calls[0]
    assert (record.outcome, record.attempts, record.retry_count, record.retry_wait_s) == ("rate_limit", 1, 0, 0.0)
    assert record.rate_limit_events[0]["action"] == "gave_up_wait_over_cap" and groq.sleeps == []


def test_running_out_of_retries_is_recorded(groq, monkeypatch):
    monkeypatch.setenv("GROQ_MAX_RETRIES", "1")
    groq(Limited(retry_after=1), Limited(retry_after=1))
    with pytest.raises(BackendError) as caught:
        talk(["hi"], scenario=None)
    record = caught.value.logs[0].llm_calls[0]
    assert (record.outcome, record.attempts, record.retry_count) == ("rate_limit", 2, 1)
    assert [e["action"] for e in record.rate_limit_events] == ["retry", "gave_up_out_of_retries"]


def test_other_errors_are_recorded_and_still_raised(groq):
    groq(ValueError("boom"))
    with pytest.raises(ValueError):
        talk(["hi"], scenario=None)
    # run_conversation only attaches logs to BackendError; the record itself is checked directly:
    record = CallRecord(model="m")
    with pytest.raises(ValueError):
        llm_backend._create(FakeClient(ValueError("boom")), record=record, messages=[])
    assert record.outcome == "error" and record.attempts == 1


def test_recording_does_not_change_the_number_of_api_calls(groq):
    client = groq(Limited(retry_after=1), used(text("hello"), 10, 2))
    talk(["hi"], scenario=None)
    assert len(client.requests) == 2  # one 429 and one success, exactly as without telemetry


def test_summary_totals():
    a = CallRecord(model="m", prompt_tokens=10, completion_tokens=2, total_tokens=12, retry_count=1, retry_wait_s=3.5,
                   rate_limit_events=[{}])
    b = CallRecord(model="m", prompt_tokens=20, completion_tokens=3, total_tokens=23)
    assert summarize([a, b]) == {
        "model_calls": 2, "prompt_tokens": 30, "completion_tokens": 5, "total_tokens": 35,
        "calls_without_usage": 0, "retry_count": 1, "retry_wait_s": 3.5, "rate_limit_events": 1,
    }


# --- the per-case JSON file ---

CASE = next(c for c in load_cases() if c["id"].startswith("price-04"))


def test_the_telemetry_file_is_written_only_when_asked_and_only_for_groq(groq, monkeypatch, tmp_path):
    groq(used(calls(CHECK, ("get_quote", QUOTE_CAR)), 1500, 100), used(text("Q-1001 is $79."), 1600, 50),
         used(calls(("validate_quote", {"quote_id": "Q-1001"})), 1700, 40), used(text("Done."), 1800, 30))
    log, failures = run_case(CASE, "groq")
    assert write_telemetry(CASE, log, failures, "groq") is None  # EVAL_TELEMETRY_DIR is not set
    monkeypatch.setenv("EVAL_TELEMETRY_DIR", str(tmp_path / "live_runs"))
    assert write_telemetry(CASE, log, failures, "guarded") is None
    path = write_telemetry(CASE, log, failures, "groq")
    data = json.loads(path.read_text())
    assert path.parent.name == "live_runs" and path.name.startswith("price-04-no-change-books-normally-")
    assert data["model"] == "openai/gpt-oss-20b" and data["totals"]["total_tokens"] == 6820
    assert [c["total_tokens"] for c in data["llm_calls"]] == [1600, 1650, 1740, 1830]
    assert "quote_issued" in [e["event"] for e in data["events"]]
    assert [t["customer"] for t in data["turns"]][0].startswith("I want to book an oil change")


def test_a_rate_limited_run_is_tagged_external_and_keeps_its_telemetry(groq, monkeypatch, tmp_path):
    monkeypatch.setenv("EVAL_TELEMETRY_DIR", str(tmp_path))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_SECRET")
    groq(Limited(retry_after=682))
    log, failures = run_case(CASE, "groq")
    assert failures[0].startswith("[external:rate_limit]")
    (path,) = tmp_path.glob("*.json")
    data = json.loads(path.read_text())
    assert data["outcome"] == "external_failure" and data["totals"]["rate_limit_events"] == 1
    assert data["llm_calls"][0]["rate_limit_events"][0]["retry_after_s"] == 682.0
    assert "gsk_SECRET" not in path.read_text()  # neither the key nor the authorization header


def test_a_normal_run_has_no_telemetry_for_scripted_backends():
    log = run_conversation(["Book an oil change Friday at 10am."], "C001", "guarded", {})[0]
    assert log.llm_calls == []
