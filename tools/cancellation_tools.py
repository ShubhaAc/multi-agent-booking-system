from langchain_core.tools import tool
from db import cancel_booking

@tool
async def cancel_room_booking(booking_id: int) -> str:
    """Cancel an active booking by its booking ID."""
    success = await cancel_booking(booking_id)
    if success:
        return f"Booking {booking_id} has been successfully cancelled."
    return f"Could not cancel booking {booking_id}. It may already be cancelled."