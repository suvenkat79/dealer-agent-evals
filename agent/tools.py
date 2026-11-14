"""Tools the service-scheduler agent can call, backed by small JSON files.

Every agent run gets a fresh copy of the data, so a booking made in one
test case never leaks into another.
"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

PRICES_USD = {
    "oil change": 79,
    "brake inspection": 120,
    "tire rotation": 49,
    "general service": 199,
}
MODEL_SURCHARGE_USD = {"F-150": 20}
NOT_APPLICABLE = {("oil change", "Model 3")}  # electric vehicle


def load_db():
    return {
        name: json.loads((DATA_DIR / f"{name}.json").read_text())
        for name in ("customers", "vehicles", "slots")
    }


class Tools:
    def __init__(self, db=None):
        self.db = db or load_db()
        self._bookings = 0

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
            return {"date": date, "open": False, "free_slots": []}
        free = sorted(time for time, state in day.items() if state == "free")
        return {"date": date, "open": True, "free_slots": free}

    def book_appointment(self, customer_id, vehicle_id, date, time, service_type):
        day = self.db["slots"].get(date)
        if day is None or day.get(time) != "free":
            return {"status": "failed", "reason": "slot not available"}
        day[time] = "booked"
        self._bookings += 1
        return {
            "status": "confirmed",
            "confirmation_id": f"APT-{self._bookings:04d}",
            "customer_id": customer_id,
            "vehicle_id": vehicle_id,
            "date": date,
            "time": time,
            "service_type": service_type,
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
)


def _schema(name, description, properties):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {k: {"type": "string", "description": v} for k, v in properties.items()},
                "required": list(properties),
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
        "Book a free slot. Fails if the slot is not free.",
        {
            "customer_id": "Customer id",
            "vehicle_id": "Vehicle id",
            "date": "Date as YYYY-MM-DD",
            "time": "Time as HH:MM, 24-hour",
            "service_type": "Service name",
        },
    ),
    _schema(
        "estimate_cost",
        "Get the price of a service for a vehicle model.",
        {"service_type": "Service name", "vehicle_model": "Vehicle model"},
    ),
]
