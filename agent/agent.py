"""Entry point: run one customer message through the agent and return a log.

Backends:
  guarded  rule-based agent that follows the dealership rules (no API key needed)
  naive    rule-based agent with realistic flaws, so the evals have bugs to catch
  groq     a real LLM with tool calling (needs GROQ_API_KEY)
"""
from dataclasses import asdict, dataclass, field

from .errors import BackendError
from .tools import Tools

BACKENDS = ("guarded", "naive", "groq")


@dataclass
class RunLog:
    message: str
    customer_id: str
    backend: str
    tool_calls: list = field(default_factory=list)
    reply: str = ""
    turns: list = field(default_factory=list)  # one RunLog per customer message in a conversation
    llm_calls: list = field(default_factory=list)  # one telemetry.CallRecord per model call (Groq backend)
    events: list = field(default_factory=list)  # quote workflow audit events raised during this turn

    def to_dict(self):
        return asdict(self)


class LoggedTools:
    """Wraps Tools so every call is recorded in the run log."""

    def __init__(self, tools, log):
        self._tools = tools
        self._log = log
        self.db = tools.db
        self.quote_state = tools.quote_state
        self.quote_workflow = tools.quote_workflow

    @property
    def turn(self):
        return self._tools.turn

    def reject(self, name, args, result):
        """Record a call that was not executed, so the run log still shows what the model attempted."""
        self._log.tool_calls.append({"name": name, "args": dict(args), "result": result})
        return result

    def call(self, name, args):
        result = self._tools.call(name, args)
        self._log.tool_calls.append({"name": name, "args": dict(args), "result": result})
        return result


def _reply(message, customer_id, backend, tools, log, session=None, history=None):
    if backend == "groq":
        from .llm_backend import run_groq

        # run_groq appends this turn's full message trail (tool calls and results included) to history.
        return run_groq(message, customer_id, tools, history=history, quoting=tools.quote_workflow, telemetry=log.llm_calls)
    from .scripted import run_scripted

    return run_scripted(message, customer_id, tools, variant=backend, session=session)


def _check_backend(backend):
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend '{backend}', expected one of {BACKENDS}")


def run_agent(message, customer_id, backend="guarded"):
    _check_backend(backend)
    log = RunLog(message=message, customer_id=customer_id, backend=backend)
    tools = LoggedTools(Tools(customer_id=customer_id), log)
    try:
        log.reply = _reply(message, customer_id, backend, tools, log)
    except BackendError as exc:
        exc.logs = [log]  # what was recorded before the API gave up, for the eval runner
        raise
    return log


def run_conversation(messages, customer_id, backend="guarded", scenario=None):
    """Run several customer messages against one shared set of tools and one agent session.

    `scenario` scripts the mock environment, e.g. a price that changes after the first quote.
    Returns one RunLog per message; each holds only that turn's tool calls and reply.
    """
    _check_backend(backend)
    tools = Tools(scenario=scenario, customer_id=customer_id)
    # The scripted quote flow keeps its memory in `session`; without a scenario there is no quote workflow.
    session = {} if tools.quote_workflow else None
    history, logs = [], []
    for number, message in enumerate(messages, start=1):
        mark = len(tools.events)
        tools.begin_turn(number, message)
        log = RunLog(message=message, customer_id=customer_id, backend=backend)
        try:
            log.reply = _reply(message, customer_id, backend, LoggedTools(tools, log), log, session=session, history=history)
        except BackendError as exc:
            log.events = tools.events_since(mark)
            exc.logs = [*logs, log]  # the turns so far, so a rate-limited run still shows what happened
            raise
        tools.end_turn(log.reply)
        log.events = tools.events_since(mark)
        if backend != "groq":
            history += [{"role": "user", "content": message}, {"role": "assistant", "content": log.reply}]
        logs.append(log)
    return logs
