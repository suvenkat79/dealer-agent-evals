"""Shared helpers for tests that call the quote tools directly."""

SLOT = dict(customer_id="C001", vehicle_id="V001", date="2026-11-20", time="10:00", service_type="oil change")
MODEL = "Civic"


def quote(tools, **override):
    """What a correct agent does: check availability for the slot, then quote that slot."""
    args = {**SLOT, **override}
    tools.check_availability(args["date"], args["service_type"])
    return tools.get_quote(
        args["service_type"], override.get("vehicle_model", MODEL), args["date"], args["time"], args["customer_id"], args["vehicle_id"]
    )
