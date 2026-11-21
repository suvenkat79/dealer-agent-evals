"""Rule-based agents that need no API key.

`guarded` follows the dealership rules. `naive` has the kind of flaws a
poorly prompted LLM agent shows: it books without checking availability,
guesses missing details, invents prices, shares other customers' records
and obeys instructions hidden in data. The eval suite should pass the
first and catch the second.
"""
import re
from datetime import date, timedelta
from difflib import SequenceMatcher

TODAY = date(2026, 11, 13)  # fixed so every run is reproducible
WEEKDAYS = {
    "monday": "2026-11-16",
    "tuesday": "2026-11-17",
    "wednesday": "2026-11-18",
    "thursday": "2026-11-19",
    "friday": "2026-11-20",
    "saturday": "2026-11-21",
    "sunday": "2026-11-22",
}
SERVICES = ("oil change", "brake inspection", "tire rotation", "general service")
SERVICE_ALIASES = {
    "brake": "brake inspection",
    "brakes": "brake inspection",
    "tire": "tire rotation",
    "tires": "tire rotation",
    "tyre": "tire rotation",
    "tyres": "tire rotation",
}


def _words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def _fuzzy_phrase(words, phrase, cutoff=0.8):
    size = len(phrase.split())
    for i in range(len(words) - size + 1):
        window = " ".join(words[i : i + size])
        if SequenceMatcher(None, window, phrase).ratio() >= cutoff:
            return True
    return False


def parse_service(text):
    words = _words(text)
    for service in SERVICES:
        if _fuzzy_phrase(words, service):
            return service
    for word in words:
        if word in SERVICE_ALIASES:
            return SERVICE_ALIASES[word]
    return None


def parse_date(text):
    iso = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if iso:
        return iso.group(1)
    words = _words(text)
    if "tomorrow" in words:
        return (TODAY + timedelta(days=1)).isoformat()
    # Exact weekday names win; only then fall back to the closest misspelling.
    # (Fuzzy-first once matched "thursday" to "tuesday"; the evals caught it.)
    for name, iso_date in WEEKDAYS.items():
        if name in words:
            return iso_date
    best_ratio, best_date = 0.0, None
    for name, iso_date in WEEKDAYS.items():
        for word in words:
            ratio = SequenceMatcher(None, word, name).ratio()
            if ratio > best_ratio:
                best_ratio, best_date = ratio, iso_date
    return best_date if best_ratio >= 0.8 else None


def parse_time(text):
    match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", text.lower())
    if not match:
        return None
    hour = int(match.group(1)) % 12
    if match.group(3) == "pm":
        hour += 12
    return f"{hour:02d}:{match.group(2) or '00'}"


def _own_vehicle(db, customer_id):
    for vehicle in db["vehicles"]:
        if vehicle["owner_id"] == customer_id:
            return vehicle
    return None


def _vehicle_by_id(db, vehicle_id):
    for vehicle in db["vehicles"]:
        if vehicle["id"] == vehicle_id:
            return vehicle
    return None


def _mentioned_vehicle_id(text):
    match = re.search(r"\b(V\d{3})\b", text.upper())
    return match.group(1) if match else None


def run_scripted(message, customer_id, tools, variant, session=None):
    guarded = variant == "guarded"
    text = message.lower()
    if session is not None:  # a multi-turn conversation with quotes
        return _quote_flow(message, customer_id, tools, guarded, session)
    own = _own_vehicle(tools.db, customer_id)
    mentioned_id = _mentioned_vehicle_id(message)
    target_id = mentioned_id or (own["id"] if own else None)
    target = _vehicle_by_id(tools.db, target_id) if target_id else None
    owns_target = bool(target and target["owner_id"] == customer_id)

    if re.search(r"\b(book|schedule|appointment|reserve)\b", text):
        return _booking(message, customer_id, tools, guarded, target_id, owns_target)
    if re.search(r"history|service record", text):
        return _history(tools, guarded, target_id, owns_target)
    if re.search(r"how much|cost|price|estimate", text):
        return _cost(message, tools, guarded, own)
    return "I can help with booking a service, cost estimates, and your vehicle's service history."


def _booking(message, customer_id, tools, guarded, vehicle_id, owns_target):
    service = parse_service(message)
    day = parse_date(message)
    time = parse_time(message)

    if not guarded:
        # Flaws: guesses missing details, skips the availability check,
        # books any vehicle, and reports success whatever the tool returns.
        service = service or "general service"
        day = day or WEEKDAYS["monday"]
        time = time or "10:00"
        tools.call(
            "book_appointment",
            {
                "customer_id": customer_id,
                "vehicle_id": vehicle_id,
                "date": day,
                "time": time,
                "service_type": service,
            },
        )
        return f"Your {service} is confirmed for {day} at {time}."

    if not owns_target:
        return "I can only book appointments for vehicles registered to you."
    missing = [
        label
        for label, value in (("the service type", service), ("the day", day), ("the time", time))
        if not value
    ]
    if missing:
        return "To book your appointment I still need " + " and ".join(missing) + "."

    availability = tools.call("check_availability", {"date": day, "service_type": service})
    if not availability["open"]:
        return f"We're closed on {day}. Please choose a weekday."
    free = availability["free_slots"]
    if time not in free:
        if not free:
            return f"{day} is fully booked. Would another day work for you?"
        return f"{time} isn't available on {day}. Free slots that day: {', '.join(free)}. Which of these would you like me to book?"

    result = tools.call(
        "book_appointment",
        {
            "customer_id": customer_id,
            "vehicle_id": vehicle_id,
            "date": day,
            "time": time,
            "service_type": service,
        },
    )
    if result.get("status") != "confirmed":
        return "Sorry, I couldn't complete that booking. Please try another slot."
    return (
        f"Your {service} is confirmed for {day} at {time}. "
        f"Appointment ID: {result['confirmation_id']}."
    )


def _history(tools, guarded, vehicle_id, owns_target):
    if guarded and not owns_target:
        return "I can only share service history for vehicles registered to you. I'm happy to pull up the history for your own vehicle if you'd like."
    record = tools.call("get_vehicle_history", {"vehicle_id": vehicle_id})
    if "error" in record:
        return "I couldn't find that vehicle."
    lines = [f"{h['date']}: {h['service']} (${h['cost_usd']})" for h in record["history"]]
    reply = "Service history: " + ("; ".join(lines) if lines else "no past services on record") + "."
    if not guarded and record["notes"]:
        # Flaw: treats free-text notes as instructions.
        reply += f" Note: {record['notes']}"
        if "free service" in record["notes"].lower():
            reply += " A free service has been applied to your account."
    return reply


def _cost(message, tools, guarded, own):
    service = parse_service(message)
    if service is None:
        if guarded:
            return "Which service would you like an estimate for?"
        return "It will be around $150."  # flaw: invented price
    result = tools.call("estimate_cost", {"service_type": service, "vehicle_model": own["model"]})
    if "error" in result:
        if guarded:
            return f"A {service} doesn't apply to your {own['model']}, so there's no charge to quote. Would you like an estimate for a different service?"
        return "That should be about $80."  # flaw: invented price
    return f"Estimated cost for a {service} on your {own['model']}: ${result['price_usd']}."


# --- Quote flow: a booking is quoted first, and the price can change before it is booked ---

from .consent import HEDGES, NO_WORDS, YES_WORDS  # noqa: E402


def _quote_flow(message, customer_id, tools, guarded, session):
    own = _own_vehicle(tools.db, customer_id)
    awaiting = session.get("awaiting")
    if awaiting:
        if re.search(NO_WORDS, message.lower()):
            return _decline(session)
        if re.search(YES_WORDS, message.lower()) and not re.search(HEDGES, message.lower()):
            return _accept(customer_id, tools, guarded, session, own)
        return f"Please reply yes to book at ${session['price']}, or no to leave it for now."

    if not re.search(r"\b(book|schedule|appointment|reserve)\b", message.lower()):
        return "I can help with booking a service, cost estimates, and your vehicle's service history."
    service, day, time = parse_service(message), parse_date(message), parse_time(message)
    missing = [l for l, v in (("the service type", service), ("the day", day), ("the time", time)) if not v]
    if missing:
        return "To book your appointment I still need " + " and ".join(missing) + "."
    availability = tools.call("check_availability", {"date": day, "service_type": service})
    if not availability["open"]:
        return f"We're closed on {day}. Please choose a weekday."
    if time not in availability["free_slots"]:
        return f"{time} isn't available on {day}. Free slots that day: {', '.join(availability['free_slots'])}."
    quote = tools.call(
        "get_quote",
        {
            "service_type": service, "vehicle_model": own["model"], "date": day, "time": time,
            "customer_id": customer_id, "vehicle_id": own["id"],
        },
    )
    if "error" in quote:
        return f"A {service} doesn't apply to your {own['model']}, so there's nothing to quote."
    session.update(
        awaiting="accept_quote", service=service, day=day, time=time, vehicle_id=own["id"],
        quote_id=quote["quote_id"], price=quote["price_usd"],
    )
    return (
        f"A {service} on {day} at {time} is quoted at ${quote['price_usd']} (quote {quote['quote_id']}). "
        "Would you like me to book it at that price?"
    )


def _book(customer_id, tools, session):
    result = tools.call(
        "book_appointment",
        {
            "customer_id": customer_id, "vehicle_id": session["vehicle_id"], "date": session["day"],
            "time": session["time"], "service_type": session["service"],
            "quote_id": session["quote_id"], "confirmed_price_usd": str(session["price"]),
        },
    )
    if result.get("status") != "confirmed":
        return None
    session.clear()
    return (
        f"Your {result['service_type']} is confirmed for {result['date']} at {result['time']} "
        f"at ${result['price_usd']}. Appointment ID: {result['confirmation_id']}."
    )


def _revalidate(tools, session):
    """Validate the session's quote. Returns None when it is bookable, else the price-change reply.

    On a change the session keeps the original quote id and price and switches to the
    replacement quote, which is the one booked if the customer accepts.
    """
    check = tools.call("validate_quote", {"quote_id": session["quote_id"]})
    if check["status"] == "valid":
        return None
    session.update(
        awaiting="accept_new_price",
        old_quote_id=session.get("old_quote_id", check["quote_id"]),
        old_price=session.get("old_price", check["quoted_price_usd"]),
        quote_id=check["replacement_quote_id"],
        price=check["current_price_usd"],
    )
    return (
        f"The price for your {session['service']} changed. Original quote {check['quote_id']} "
        f"(${check['quoted_price_usd']}) is no longer valid. The updated price is ${check['current_price_usd']} "
        f"(+${check['price_change_usd']}) under replacement quote {check['replacement_quote_id']}. "
        f"I haven't booked anything. Do you want to book at ${check['current_price_usd']}?"
    )


def _accept(customer_id, tools, guarded, session, own):
    if not guarded:
        # Flaw: books without validating first. When the booking is refused it validates, switches to
        # the replacement quote without asking the customer and tries again, then claims success.
        reply = _book(customer_id, tools, session)
        if reply:
            return reply
        check = tools.call("validate_quote", {"quote_id": session["quote_id"]})
        if "replacement_quote_id" in check:
            session.update(quote_id=check["replacement_quote_id"], price=check["current_price_usd"])
        return _book(customer_id, tools, session) or "Your appointment is confirmed."

    # Validation comes before every booking, including after the customer accepts a new price,
    # because the price could move again. The validated quote id is the one that gets booked.
    changed = _revalidate(tools, session)
    if changed:
        return changed
    return _book(customer_id, tools, session) or "Sorry, I couldn't complete that booking."


def _decline(session):
    if session["awaiting"] == "accept_new_price":
        reply = (
            f"Understood. Quote {session['old_quote_id']} (${session['old_price']}) is no longer valid because the price "
            f"changed to ${session['price']}, so no appointment has been booked. "
            "Tell me if you'd like to book at the new price or try another service."
        )
    else:
        reply = "No problem, I haven't booked anything. Let me know if you'd like another time or service."
    session.clear()
    return reply
