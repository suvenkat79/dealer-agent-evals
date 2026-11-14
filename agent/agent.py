"""Entry point: run one customer message through the agent and return a log.

Backends:
  guarded  rule-based agent that follows the dealership rules (no API key needed)
  naive    rule-based agent with realistic flaws, so the evals have bugs to catch
  groq     a real LLM with tool calling (needs GROQ_API_KEY)
"""
from dataclasses import asdict, dataclass, field

from .tools import Tools

BACKENDS = ("guarded", "naive", "groq")


@dataclass
class RunLog:
    message: str
    customer_id: str
    backend: str
    tool_calls: list = field(default_factory=list)
    reply: str = ""

    def to_dict(self):
        return asdict(self)


class LoggedTools:
    """Wraps Tools so every call is recorded in the run log."""

    def __init__(self, tools, log):
        self._tools = tools
        self._log = log
        self.db = tools.db

    def call(self, name, args):
        result = self._tools.call(name, args)
        self._log.tool_calls.append({"name": name, "args": dict(args), "result": result})
        return result


def run_agent(message, customer_id, backend="guarded"):
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend '{backend}', expected one of {BACKENDS}")
    log = RunLog(message=message, customer_id=customer_id, backend=backend)
    tools = LoggedTools(Tools(), log)
    if backend == "groq":
        from .llm_backend import run_groq

        log.reply = run_groq(message, customer_id, tools)
    else:
        from .scripted import run_scripted

        log.reply = run_scripted(message, customer_id, tools, variant=backend)
    return log
