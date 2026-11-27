"""Mock booking tools: reservations are just documents in MongoDB."""
from datetime import datetime, timezone

from src.db import bookings


def _overlap(stay_id: str, check_in: str, check_out: str) -> dict:
    """Query for an existing booking that overlaps the requested dates."""
    return {"stay_id": stay_id, "check_in": {"$lt": check_out}, "check_out": {"$gt": check_in}}


def check_availability(stay_id: str, check_in: str, check_out: str) -> dict:
    """Check if a stay is free between two dates (format YYYY-MM-DD)."""
    taken = bookings.find_one(_overlap(stay_id, check_in, check_out))
    return {"available": taken is None}


def book_stay(stay_name: str, stay_id: str, check_in: str, check_out: str) -> dict:
    """Book a stay (dates as YYYY-MM-DD). The system asks the user to confirm first."""
    if bookings.find_one(_overlap(stay_id, check_in, check_out)):
        return {"status": "unavailable"}
    bookings.insert_one(
        {
            "stay_id": stay_id,
            "stay_name": stay_name,
            "check_in": check_in,
            "check_out": check_out,
            "created_at": datetime.now(timezone.utc),
        }
    )
    return {"status": "confirmed"}
