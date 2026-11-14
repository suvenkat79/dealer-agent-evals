"""Run every eval case against one agent backend.

    AGENT_BACKEND=guarded pytest      # default, no API key needed
    AGENT_BACKEND=naive pytest        # the flawed agent: expect failures
    AGENT_BACKEND=groq EVAL_RUNS=3 pytest   # real LLM, each case run 3 times
"""
import os

import pytest

from evals.runner import load_cases, run_case

BACKEND = os.getenv("AGENT_BACKEND", "guarded")
RUNS = int(os.getenv("EVAL_RUNS", "1"))
CASES = load_cases()


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_case(case):
    problems = []
    for attempt in range(1, RUNS + 1):
        log, failures = run_case(case, BACKEND)
        if failures:
            problems.append(f"run {attempt}: {failures}\n  reply: {log.reply}\n  calls: {log.tool_calls}")
    assert not problems, f"{len(problems)}/{RUNS} runs failed\n" + "\n".join(problems)
