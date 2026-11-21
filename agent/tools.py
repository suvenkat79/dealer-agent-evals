"""Tools the service-scheduler agent can call, backed by small JSON files.

Every agent run gets a fresh copy of the data, so a booking made in one
test case never leaks into another.
"""
import json
import re
import unicodedata
from pathlib import Path

from .consent import classify_reply

DATA_DIR = Path(__file__).parent / "data"

PRICES_USD = {
    "oil change": 79,
    "brake inspection": 120,
    "tire rotation": 49,
    "general service": 199,
}
MODEL_SURCHARGE_USD = {"F-150": 20}
NOT_APPLICABLE = {("oil change", "Model 3")}  # electric vehicle


# LLMs sometimes write quote ids with a non-breaking hyphen or full-width digits.
# Only the dashes and character width are normalised; the digits must still match exactly.
_ID_DASH = re.compile(r"([A-Za-z]{1,4})[\u2010-\u2015\u2212\ufe58\ufe63\uff0d]+(?=\d)")


def normalize_id(value):
    """Q\u20111001 -> Q-1001. 'Q-1003' stays 'Q-1003': similar-looking ids are never merged."""
    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub("[\u200b\u200c\u200d\u2060\ufeff]", "", text).strip()
    return _ID_DASH.sub(r"\1-", text)


SLOT_KEYS = ("customer_id", "vehicle_id", "date", "time")
# What a quote is bound to, and the failure reason when a booking asks for something else.
QUOTE_BINDINGS = (
    ("service_type", "quote_service_mismatch"),
    ("customer_id", "quote_customer_mismatch"),
    ("vehicle_id", "quote_vehicle_mismatch"),
    ("date", "quote_slot_mismatch"),
    ("time", "quote_slot_mismatch"),
)


def _slot(customer_id, vehicle_id, date, time):
    return {"customer_id": customer_id, "vehicle_id": vehicle_id, "date": date, "time": time}


def load_db():
    return {
        name: json.loads((DATA_DIR / f"{name}.json").read_text())
        for name in ("customers", "vehicles", "slots")
    }


class Tools:
    def __init__(self, db=None, scenario=None):
        self.db = db or load_db()
        self._bookings = 0
        self.quotes = {}
        self.validated = set()  # quote ids that validate_quote has checked; booking requires one
        self.original_quote_id = None  # the first quote issued; set once, never overwritten
        self._checked = set()  # quote ids validate_quote has already been called on
        # Optional scripted price change, e.g.
        # {"price_change": {"service_type": "oil change", "original_usd": 150, "updated_usd": 180}}.
        # The price flips right after the first quote for that service is issued, so tests
        # never depend on live pricing.
        self._change = (scenario or {}).get("price_change")
        # A scenario (even an empty one) means this conversation runs the quote-then-book workflow.
        self.quote_workflow = scenario is not None
        # Customer turn counter, set by the conversation runner through begin_turn(). None means turns
        # are not tracked (direct tool use), which switches off the customer-consent workflow below.
        self.turn = None
        self._turn_decision = None
        # Per quote: when it was shown to the customer, what they answered, and when it was validated.
        self.workflow = {}
        self._price_changed = False
        # Free slots seen by check_availability, per (service, date). A quote or a booking is only
        # allowed for a slot that appeared in such a result.
        self._availability = {}
        # Append-only audit trail of the quote workflow: what was shown, answered, validated and booked.
        # Events hold ids, prices, turns and reason codes, never the customer's message text.
        self.events = []

    def get_vehicle_history(self, vehicle_id):
        for vehicle in self.db["vehicles"]:
            if vehicle["id"] == vehicle_id:
                return {
                    "vehicle_id": vehicle["id"],
                    "owner_id": vehicle["owner_id"],
                    "model": vehicle["model"],
                    "history": vehicle["history"],
                    "notes": vehicle["notes"],
                }
        return {"error": f"vehicle {vehicle_id} not found"}

    def check_availability(self, date, service_type):
        day = self.db["slots"].get(date)
        if day is None:
            self._availability[(service_type, date)] = set()
            return {"date": date, "open": False, "free_slots": []}
        free = sorted(time for time, state in day.items() if state == "free")
        self._availability[(service_type, date)] = set(free)
        return {"date": date, "open": True, "free_slots": free}

    def _emit(self, event, **fields):
        self.events.append({"seq": len(self.events) + 1, "turn": self.turn, "event": event, **fields})

    def events_since(self, mark):
        """Copies of the events after the first `mark`, so a caller cannot alter the trail."""
        return [dict(event) for event in self.events[mark:]]

    def _current_price(self, service_type, vehicle_model):
        if self._change and service_type == self._change["service_type"]:
            return self._change["updated_usd" if self._price_changed else "original_usd"]
        return PRICES_USD[service_type] + MODEL_SURCHARGE_USD.get(vehicle_model, 0)

    def _quote_view(self, quote_id):
        quote = self.quotes[quote_id]
        current = self._current_price(quote["service_type"], quote["vehicle_model"])
        return {
            "quote_id": quote_id,
            "service_type": quote["service_type"],
            "status": "valid" if current == quote["price_usd"] else "expired",
            "quoted_price_usd": quote["price_usd"],
            "current_price_usd": current,
            "price_change_usd": current - quote["price_usd"],
        }

    def _issue_quote(self, service_type, vehicle_model, slot):
        """Return a valid quote at today's price, reusing an existing one so ids are never wasted.

        `slot` binds the quote to a customer, vehicle, date and time: a quote only covers that booking.
        """
        price = self._current_price(service_type, vehicle_model)
        wanted = {"service_type": service_type, "vehicle_model": vehicle_model, "price_usd": price, **slot}
        for quote_id, quote in self.quotes.items():
            if quote == wanted:
                return quote_id
        quote_id = f"Q-{1001 + len(self.quotes)}"
        self.quotes[quote_id] = wanted
        if self.original_quote_id is None:
            self.original_quote_id = quote_id
        self.workflow[quote_id] = {
            "presented_turn": None, "decision": None, "decided_turn": None, "validated_turn": None,
            "booked_turn": None,
        }
        if self._change and service_type == self._change["service_type"]:
            self._price_changed = True  # the price moves after the first quote goes out
        return quote_id

    def _original_fields(self):
        if self.original_quote_id is None:
            return {}
        return {
            "original_quote_id": self.original_quote_id,
            "original_price_usd": self.quotes[self.original_quote_id]["price_usd"],
        }

    def quote_state(self):
        """Authoritative quote state, for the agent to read. The agent cannot write it."""
        return {
            **self._original_fields(),
            "quotes": {
                quote_id: {
                    "price_usd": quote["price_usd"],
                    "status": self._quote_view(quote_id)["status"],
                    "validated": quote_id in self.validated,
                    "presented_turn": self.workflow[quote_id]["presented_turn"],
                    "customer_decision": self.workflow[quote_id]["decision"],
                }
                for quote_id, quote in self.quotes.items()
            },
        }

    def begin_turn(self, number, message):
        """Start a customer turn and read the message as an answer to the quote on the table.

        The quote it answers is the latest one shown to the customer in an earlier turn that has
        no answer yet. Only a clear yes at that quote's price counts as acceptance.
        """
        self.turn, self._turn_decision = number, None
        open_quotes = [
            (flow["presented_turn"], quote_id)
            for quote_id, flow in self.workflow.items()
            if flow["presented_turn"] is not None and flow["presented_turn"] < number and flow["decision"] is None
        ]
        if open_quotes:
            quote_id = max(open_quotes)[1]
            price = self.quotes[quote_id]["price_usd"]
            decision = classify_reply(message, price)
            if decision:
                self.workflow[quote_id].update(decision=decision, decided_turn=number)
                self._turn_decision = decision
                self._emit(f"customer_{decision}", quote_id=quote_id, price_usd=price)
            else:
                self._emit("customer_response_unclear", quote_id=quote_id, price_usd=price)
        if self._turn_decision is None:
            self._withdraw_acceptance(number, message)

    def _withdraw_acceptance(self, number, message):
        """A customer who said yes can still change their mind before the booking is made.

        An accepted quote that is not booked yet is turned into a rejection when a later message
        declines. It can only be booked again after the customer is shown it again and re-accepts.
        """
        for quote_id, flow in self.workflow.items():
            if flow["decision"] != "accepted" or flow["booked_turn"] is not None or flow["decided_turn"] >= number:
                continue
            if classify_reply(message, self.quotes[quote_id]["price_usd"]) == "rejected":
                flow.update(decision="rejected", decided_turn=number)
                self._turn_decision = "rejected"
                self._emit("customer_acceptance_withdrawn", quote_id=quote_id, price_usd=self.quotes[quote_id]["price_usd"])

    def end_turn(self, reply):
        """A quote counts as shown to the customer once the reply names both its id and its price."""
        text = normalize_id(reply)
        for quote_id, quote in self.quotes.items():
            flow = self.workflow[quote_id]
            names_id = re.search(rf"\b{re.escape(quote_id)}\b", text)
            names_price = re.search(rf"\$\s?{quote['price_usd']}(?:\.00?)?(?!\d)", text)
            reopened = flow["decision"] == "rejected" and flow["decided_turn"] < self.turn
            if names_id and names_price and (flow["presented_turn"] is None or reopened):
                flow.update(presented_turn=self.turn, decision=None, decided_turn=None)
                self._emit("quote_presented", quote_id=quote_id, price_usd=quote["price_usd"], reopened=bool(reopened))

    def _consent_block(self, quote_id, for_booking):
        """Why the customer's consent does not yet allow validating or booking this quote, or None."""
        flow = self.workflow[quote_id]
        price = self.quotes[quote_id]["price_usd"]
        if self._turn_decision == "rejected" or flow["decision"] == "rejected":
            return "customer_rejected", "The customer declined. Do not book. Tell them no appointment was booked."
        if flow["presented_turn"] is None or flow["presented_turn"] >= self.turn:
            return "quote_not_presented", f"Tell the customer quote {quote_id} and its price ${price} in your reply, then wait for their answer in a later message."
        if flow["decision"] != "accepted":
            return "customer_acceptance_required", f"The customer has not clearly accepted quote {quote_id} at ${price}. Ask them to confirm."
        if for_booking and flow["validated_turn"] is None:
            return "validation_required", f"Call validate_quote on {quote_id} (on its own, before booking) now that the customer accepted it."
        return None

    def _slot_not_checked(self, service_type, date, time):
        """Why check_availability does not cover this service, date and time, or None."""
        free = self._availability.get((service_type, date))
        if free is None:
            return {
                "reason": "availability_not_checked",
                "detail": f"Call check_availability for {service_type} on {date} first.",
            }
        if time not in free:
            return {
                "reason": "slot_not_available",
                "detail": f"{time} was not listed as free for {service_type} on {date}. Tell the customer and ask for another time.",
            }
        return None

    def get_quote(self, service_type, vehicle_model, date, time, customer_id, vehicle_id):
        known = set(self.quotes)
        result = self._get_quote(service_type, vehicle_model, date, time, customer_id, vehicle_id)
        if "error" in result:
            self._emit("action_refused", action="get_quote", reason=result.get("reason", "error"))
        else:
            self._emit(
                "quote_issued", quote_id=result["quote_id"], price_usd=result["price_usd"], date=date, time=time,
                reused=result["quote_id"] in known,
            )
        return result

    def _get_quote(self, service_type, vehicle_model, date, time, customer_id, vehicle_id):
        """Issue a quote at today's price for one slot. It stays valid only while that price holds.

        Needs check_availability to have listed `time` as free for this service and date, and the
        vehicle to belong to the customer. The quote is bound to all of these.
        """
        priced = self.estimate_cost(service_type, vehicle_model)
        if "error" in priced:
            return priced
        vehicle = next((v for v in self.db["vehicles"] if v["id"] == vehicle_id), None)
        if vehicle is None:
            return {"error": f"vehicle {vehicle_id} not found", "reason": "vehicle_not_found"}
        if vehicle["owner_id"] != customer_id:
            return {"error": f"vehicle {vehicle_id} is not registered to {customer_id}", "reason": "vehicle_customer_mismatch"}
        if vehicle["model"] != vehicle_model:
            return {"error": f"vehicle {vehicle_id} is a {vehicle['model']}, not a {vehicle_model}", "reason": "vehicle_model_mismatch"}
        unchecked = self._slot_not_checked(service_type, date, time)
        if unchecked:
            return {"error": "cannot quote yet", **unchecked}
        original = self.original_quote_id
        if original and original not in self._checked and self._quote_view(original)["status"] == "expired":
            # The original quote must be validated first; its validation supplies the replacement.
            return {"error": f"quote {original} is still outstanding: call validate_quote on {original} first"}
        quote_id = self._issue_quote(service_type, vehicle_model, _slot(customer_id, vehicle_id, date, time))
        return {
            "quote_id": quote_id,
            "service_type": service_type,
            "date": date,
            "time": time,
            "price_usd": self.quotes[quote_id]["price_usd"],
            "status": "valid",
        }

    def validate_quote(self, quote_id):
        result = self._validate_quote(quote_id)
        if "error" in result:
            self._emit("action_refused", action="validate_quote", reason=result.get("reason", "error"))
        elif result["status"] == "valid":
            self._emit("quote_validated", quote_id=result["quote_id"], price_usd=result["current_price_usd"])
        else:
            self._emit(
                "quote_expired", quote_id=result["quote_id"], quoted_price_usd=result["quoted_price_usd"],
                current_price_usd=result["current_price_usd"], replacement_quote_id=result["replacement_quote_id"],
            )
        return result

    def _validate_quote(self, quote_id):
        """Check a quote against the current price.

        A valid quote is marked validated and can be booked. An expired one reports the price
        change and comes with a replacement quote at the current price, already validated, so the
        agent never needs to request another one. The original quote must be validated first:
        validating a replacement does not stand in for it. Repeat calls are answered from state
        and flagged `already_validated`.
        """
        quote_id = normalize_id(quote_id)
        if quote_id not in self.quotes:
            return {"error": f"quote {quote_id} not found"}
        if self.original_quote_id not in self._checked and quote_id != self.original_quote_id:
            return {"error": f"validate the original quote {self.original_quote_id} first"}
        if self.turn is not None:
            blocked = self._consent_block(quote_id, for_booking=False)
            if blocked:
                return {"error": f"cannot validate {quote_id} yet", "reason": blocked[0], "detail": blocked[1]}
        repeat = quote_id in self._checked
        self._checked.add(quote_id)
        view = {**self._quote_view(quote_id), **self._original_fields(), **({"already_validated": True} if repeat else {})}
        if view["status"] == "valid":
            self.validated.add(quote_id)
            self.workflow[quote_id]["validated_turn"] = self.turn
            return view
        quote = self.quotes[quote_id]
        replacement = self._issue_quote(quote["service_type"], quote["vehicle_model"], _slot(*(quote[k] for k in SLOT_KEYS)))
        self.validated.add(replacement)
        self._checked.add(replacement)
        return {**view, "replacement_quote_id": replacement}

    def book_appointment(self, customer_id, vehicle_id, date, time, service_type, quote_id=None, confirmed_price_usd=None):
        result = self._book_appointment(customer_id, vehicle_id, date, time, service_type, quote_id, confirmed_price_usd)
        if result.get("status") == "confirmed":
            self._emit(
                "booking_confirmed", confirmation_id=result["confirmation_id"], quote_id=result.get("quote_id"),
                price_usd=result.get("price_usd"), service_type=service_type, date=date, time=time,
            )
        else:
            self._emit("action_refused", action="book_appointment", reason=result.get("reason", "error"))
        return result

    def _book_appointment(self, customer_id, vehicle_id, date, time, service_type, quote_id=None, confirmed_price_usd=None):
        if self.quote_workflow and quote_id is None:
            return {
                "status": "failed",
                "reason": "quote_required",
                "detail": "Call check_availability, then get_quote, show the customer the quote id and price, and wait for their answer.",
            }
        if quote_id is not None:
            quote_id = normalize_id(quote_id)
            if quote_id not in self.quotes:
                return {"status": "failed", "reason": "unknown_quote"}
            view = self._quote_view(quote_id)
            if quote_id not in self.validated:
                return {
                    "status": "failed",
                    "reason": "validation_required",
                    "detail": f"Call validate_quote on {quote_id} (on its own, before booking) first.",
                }
            booked = {"service_type": service_type, "customer_id": customer_id, "vehicle_id": vehicle_id, "date": date, "time": time}
            for key, reason in QUOTE_BINDINGS:
                if self.quotes[quote_id][key] != booked[key]:
                    return {
                        "status": "failed",
                        "reason": reason,
                        "detail": f"Quote {quote_id} is for {key} {self.quotes[quote_id][key]!r}, not {booked[key]!r}. Request a quote for what the customer wants.",
                    }
            if quote_id != self.original_quote_id:
                # A replacement carries a price the customer has not agreed to; they must state it.
                if confirmed_price_usd is None:
                    return {"status": "failed", "reason": "price_confirmation_required"}
            if view["status"] != "valid":
                return {
                    "status": "failed",
                    "reason": "quote_expired",
                    "quoted_price_usd": view["quoted_price_usd"],
                    "current_price_usd": view["current_price_usd"],
                }
            if confirmed_price_usd is not None and float(confirmed_price_usd) != view["current_price_usd"]:
                return {"status": "failed", "reason": "price_mismatch", "current_price_usd": view["current_price_usd"]}
            if self.turn is not None:
                blocked = self._consent_block(quote_id, for_booking=True)
                if blocked:
                    return {"status": "failed", "reason": blocked[0], "detail": blocked[1]}
            if (service_type, date) not in self._availability:
                return {"status": "failed", "reason": "availability_not_checked", "detail": f"Call check_availability for {service_type} on {date} first."}
        day = self.db["slots"].get(date)
        if day is None or day.get(time) != "free":
            return {"status": "failed", "reason": "slot not available"}
        day[time] = "booked"
        if quote_id:
            self.workflow[quote_id]["booked_turn"] = self.turn
        for (_, checked_date), free in self._availability.items():
            if checked_date == date:
                free.discard(time)
        self._bookings += 1
        return {
            "status": "confirmed",
            "confirmation_id": f"APT-{self._bookings:04d}",
            "customer_id": customer_id,
            "vehicle_id": vehicle_id,
            "date": date,
            "time": time,
            "service_type": service_type,
            **({"quote_id": quote_id, "price_usd": view["current_price_usd"]} if quote_id else {}),
        }

    def estimate_cost(self, service_type, vehicle_model):
        if service_type not in PRICES_USD:
            return {"error": f"unknown service '{service_type}'"}
        if (service_type, vehicle_model) in NOT_APPLICABLE:
            return {"error": f"{service_type} is not applicable to {vehicle_model}"}
        price = PRICES_USD[service_type] + MODEL_SURCHARGE_USD.get(vehicle_model, 0)
        return {
            "service_type": service_type,
            "vehicle_model": vehicle_model,
            "price_usd": price,
        }

    def call(self, name, args):
        """Dispatch a tool call by name. Bad names or arguments return an error dict."""
        if name not in TOOL_NAMES:
            return {"error": f"unknown tool '{name}'"}
        try:
            return getattr(self, name)(**args)
        except TypeError as exc:
            return {"error": f"bad arguments for {name}: {exc}"}


TOOL_NAMES = (
    "get_vehicle_history",
    "check_availability",
    "book_appointment",
    "estimate_cost",
    "get_quote",
    "validate_quote",
)


def _schema(name, description, properties, optional=()):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {k: {"type": "string", "description": v} for k, v in properties.items()},
                "required": [k for k in properties if k not in optional],
            },
        },
    }


# OpenAI-style tool schemas, used by the LLM backend.
TOOL_SCHEMAS = [
    _schema(
        "get_vehicle_history",
        "Get the service history and notes for one vehicle.",
        {"vehicle_id": "Vehicle id, for example V001"},
    ),
    _schema(
        "check_availability",
        "List free appointment slots for a date.",
        {"date": "Date as YYYY-MM-DD", "service_type": "Service name"},
    ),
    _schema(
        "book_appointment",
        "Book a free slot. Fails if the slot is not free. A quoted booking also needs a quote_id that validate_quote has checked. A replacement quote needs confirmed_price_usd. Booking is refused until the customer has been shown the quote and has accepted it in a later message.",
        {
            "customer_id": "Customer id",
            "vehicle_id": "Vehicle id",
            "date": "Date as YYYY-MM-DD",
            "time": "Time as HH:MM, 24-hour",
            "service_type": "Service name",
            "quote_id": "Quote id from get_quote. Required when booking a quoted service",
            "confirmed_price_usd": "The price in dollars the customer explicitly agreed to",
        },
        optional=("quote_id", "confirmed_price_usd"),
    ),
    _schema(
        "estimate_cost",
        "Get the price of a service for a vehicle model.",
        {"service_type": "Service name", "vehicle_model": "Vehicle model"},
    ),
    _schema(
        "get_quote",
        "Issue a price quote for one appointment slot. Call check_availability for the service and date first, and only quote a time it lists as free. The quote covers only this service, customer, vehicle, date and time. It expires if the price changes before booking.",
        {
            "service_type": "Service name",
            "vehicle_model": "Vehicle model",
            "date": "Date as YYYY-MM-DD",
            "time": "Time as HH:MM, 24-hour",
            "customer_id": "Customer id",
            "vehicle_id": "Vehicle id",
        },
    ),
    _schema(
        "validate_quote",
        "Check whether a quote is still valid. If the price changed it returns the new price, the difference and a replacement_quote_id to use from then on. Required before booking.",
        {"quote_id": "Quote id, for example Q-1001"},
    ),
]
