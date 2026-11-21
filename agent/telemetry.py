"""Per-call model telemetry: token usage, retries and rate-limit errors.

One CallRecord is written for every model call the Groq backend makes (a call that is retried after
an HTTP 429 is still one record). Records hold response fields only: never request headers, so the
API key cannot end up in them.
"""
from dataclasses import asdict, dataclass, field


@dataclass
class CallRecord:
    model: str
    turn: int | None = None  # customer turn the call belongs to
    step: int = 0  # model round trip within that turn, starting at 1
    outcome: str = "pending"  # ok | rate_limit | error
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    attempts: int = 0  # HTTP requests made, including the first
    retry_count: int = 0  # attempts - 1 when it was retried
    retry_wait_s: float = 0.0  # time slept between attempts
    elapsed_s: float = 0.0  # wall time for the whole call, retry waits included
    rate_limit_events: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


# Headers worth keeping from a 429. Nothing that could carry credentials is ever copied.
RATE_LIMIT_HEADERS = (
    "retry-after",
    "x-ratelimit-limit-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-tokens",
)


def rate_limit_headers(exc):
    headers = getattr(getattr(exc, "response", None), "headers", None)
    try:
        return {name: headers[name] for name in RATE_LIMIT_HEADERS if headers is not None and name in headers}
    except Exception:  # an odd headers object must never break a model call
        return {}


def read_usage(response):
    """(prompt, completion, total) token counts, each None when the response does not report them."""
    usage = getattr(response, "usage", None)
    return tuple(getattr(usage, name, None) for name in ("prompt_tokens", "completion_tokens", "total_tokens"))


def summarize(calls):
    """Totals over a list of CallRecords. Calls with unreported tokens are counted, not guessed at."""
    def total(name):
        values = [getattr(c, name) for c in calls if getattr(c, name) is not None]
        return sum(values) if values else None

    return {
        "model_calls": len(calls),
        "prompt_tokens": total("prompt_tokens"),
        "completion_tokens": total("completion_tokens"),
        "total_tokens": total("total_tokens"),
        "calls_without_usage": sum(c.total_tokens is None for c in calls),
        "retry_count": sum(c.retry_count for c in calls),
        "retry_wait_s": round(sum(c.retry_wait_s for c in calls), 2),
        "rate_limit_events": sum(len(c.rate_limit_events) for c in calls),
    }
