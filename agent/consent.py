"""Read a customer's reply to a quote as accepted, rejected or neither.

Deliberately conservative: anything unclear counts as neither, and neither never authorises a
booking. The customer has to say yes to the price on the table, without hedging.
"""
import re

# "No problem, go ahead" starts with "no" but is not a refusal.
NO_WORDS = r"^\s*no\b(?!\s+(problem|worries|issue))|\b(don't|do not|not (want|interested)|cancel|never ?mind|decline|too (much|expensive))\b"
YES_WORDS = r"\b(yes|yeah|yep|ok|okay|sure|fine|accept|agree|go ahead|book it|works)\b"
# A hedge, a "but" or a question means the customer has not clearly agreed ("ok maybe", "sure, but...").
HEDGES = (
    r"\b(maybe|perhaps|not sure|unsure|i guess|i suppose|i think|think about|thinking about|hold on|wait"
    r"|later|depends|however|but)\b|\?"
)


def classify_reply(message, price_usd):
    """Return "rejected", "accepted" or None for a customer message about a quote at `price_usd`."""
    text = message.lower()
    if re.search(NO_WORDS, text):
        return "rejected"
    if re.search(HEDGES, text) or not re.search(YES_WORDS, text):
        return None
    amounts = [float(a) for a in re.findall(r"\$\s?(\d+(?:\.\d+)?)", message)]
    if amounts and float(price_usd) not in amounts:
        return None  # "yes, $150" is not a yes to a quote of $180
    return "accepted"
