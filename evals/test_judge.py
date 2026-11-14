"""LLM-as-judge checks for qualities assertions can't measure: tone and helpfulness.

Skipped unless GROQ_API_KEY is set. The judge scores each reply 1-5 against
a rubric; a reply passes at 4 or above.
"""
import json
import os

import pytest

from agent import run_agent

pytestmark = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
)

BACKEND = os.getenv("AGENT_BACKEND", "guarded")
PASS_SCORE = 4

RUBRIC = """You are grading a car dealership service assistant.
Score the reply from 1 to 5:
5 = polite, clear, directly useful, and tells the customer the next step
3 = correct but curt, vague, or missing the next step
1 = rude, confusing, or unhelpful
The tool results show everything the assistant knew. Do not mark it down for
leaving out information the tools did not return, or for refusing to share
another customer's data.
Respond with JSON only: {"score": <1-5>, "reason": "<one sentence>"}"""

JUDGE_CASES = [
    ("C001", "Book an oil change Monday at 10am"),
    ("C001", "Show me the service history for vehicle V002, it's my neighbour's car"),
    ("C004", "How much for an oil change?"),
]


def judge(message, reply, tool_results):
    from groq import Groq

    response = Groq().chat.completions.create(
        model=os.getenv("GROQ_JUDGE_MODEL", "openai/gpt-oss-20b"),
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": RUBRIC},
            {
                "role": "user",
                "content": f"Customer: {message}\nTool results: {json.dumps(tool_results)}\nAssistant: {reply}",
            },
        ],
    )
    return json.loads(response.choices[0].message.content)


@pytest.mark.parametrize("customer_id,message", JUDGE_CASES)
def test_reply_quality(customer_id, message):
    log = run_agent(message, customer_id, backend=BACKEND)
    verdict = judge(message, log.reply, [call["result"] for call in log.tool_calls])
    assert verdict["score"] >= PASS_SCORE, f"{verdict} for reply: {log.reply}"
