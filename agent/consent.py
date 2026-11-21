"""Read a customer's reply to a quote as accepted, rejected or neither.

Deliberately conservative: anything unclear counts as neither, and neither never authorises a
booking. The customer has to say yes to the price on the table, without hedging, negating or
attaching a condition, and any amount they mention must be that price.
"""
import re
import unicodedata

# "No problem, go ahead" starts with "no" but is not a refusal. Apostrophes are normalised first, so
# a phone keyboard's "don\u2019t" is read like "don't".
NO_WORDS = r"^\s*no\b(?!\s+(problem|worries|issue))|\b(don't|do not|not (want|interested)|cancel|never ?mind|decline|too (much|expensive))\b"
YES_WORDS = r"\b(yes|yeah|yep|ok|okay|sure|fine|accept|agree|go ahead|book it|works)\b"
# A hedge, a condition, a "but" or a question means the customer has not clearly agreed
# ("ok maybe", "sure, if it is quick", "yes, but...").
HEDGES = (
    r"\b(maybe|perhaps|probably|possibly|might|could|not sure|unsure|i guess|i suppose|i think|think about"
    r"|thinking about|hold on|wait|later|depends|however|but|if|unless|as long as|provided|only if|assuming"
    r"|kind of|sort of|we'll see|i'll see|whatever|hmm+|um+|uh+)\b|\?"
)
# "not fine", "not okay", "that doesn't work": a yes-word under a negation is not a yes.
NEGATIONS = r"\b(not|never|cannot|nope|nah|no)\b(?!\s+(problem|worries|issue))|n't\b"

_QUOTE_OR_ID = re.compile(r"\b[a-z]{1,4}-\d+\b")  # Q-1001, APT-0001: ids are not amounts
_DATE_OR_TIME = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b|\b\d{1,2}:\d{2}\s*(?:am|pm)?(?!\w)"
    r"|\b\d{1,2}\s*(?:am|pm|a\.m\.|p\.m\.)(?!\w)|\b\d+(?:st|nd|rd|th)\b"
)
# A price spelled out ("one fifty") cannot be compared safely, so it is not a clear yes.
_SPELLED_AMOUNT = r"\b(ten|eleven|twelve|(?:thir|four|fif|six|seven|eigh|nine)teen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|grand|bucks)\b"
_NUMBER = re.compile(r"(?<![\w.])\$?\s?(\d+(?:,\d{3})*(?:\.\d+)?)(?!\w)")


def _plain(message):
    text = unicodedata.normalize("NFKC", message).lower()
    return text.translate(str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'", "`": "'"}))


def _amounts(text):
    """Every number in the reply that could be a price: with or without a $ sign, never a time or id."""
    text = _DATE_OR_TIME.sub(" ", _QUOTE_OR_ID.sub(" ", text))
    return [float(n.replace(",", "")) for n in _NUMBER.findall(text)]


def classify_reply(message, price_usd):
    """Return "rejected", "accepted" or None for a customer message about a quote at `price_usd`."""
    text = _plain(message)
    if re.search(NO_WORDS, text):
        return "rejected"
    if re.search(HEDGES, text) or re.search(NEGATIONS, text) or not re.search(YES_WORDS, text):
        return None
    if re.search(_SPELLED_AMOUNT, text):
        return None
    if any(abs(amount - float(price_usd)) > 0.005 for amount in _amounts(text)):
        return None  # "yes, 150" is not a yes to a quote of $180, with or without the $ sign
    return "accepted"
