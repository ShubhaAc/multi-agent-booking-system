from langchain_core.tools import tool
from db import check_availability, create_booking, find_alternative_room

@tool
async def check_room_availability(room_name: str, booking_date: str, booking_time: str, duration_minutes: int = 60) -> str:
    """Check if a room is available on a given date and time for a given duration."""
    available = await check_availability(room_name, booking_date, booking_time, duration_minutes)
    if available:
        return f"{room_name} is available on {booking_date} at {booking_time} for {duration_minutes} minutes."
    return f"{room_name} is not available on {booking_date} at {booking_time} for {duration_minutes} minutes."

@tool
async def book_room(room_name: str, booked_by: str, invitee_email: str, booking_date: str, booking_time: str, duration_minutes: int = 60) -> str:
    """Book a room for a given date, time and duration."""
    booking_id = await create_booking(room_name, booked_by, invitee_email, booking_date, booking_time, duration_minutes)
    return f"Booking confirmed. Booking ID is {booking_id}. Duration: {duration_minutes} minutes."

@tool
async def find_alternative(room_name: str, booking_date: str, booking_time: str, duration_minutes: int = 60) -> str:
    """Find an alternative available room when the requested room is taken."""
    alternative = await find_alternative_room(room_name, booking_date, booking_time, duration_minutes)
    if alternative:
        return f"{alternative} is available on {booking_date} at {booking_time} for {duration_minutes} minutes."
    return "No alternative rooms are available at that time."

@tool
async def list_available_rooms(booking_date: str, booking_time: str, duration_minutes: int = 60) -> str:
    """List all rooms available on a given date and time for a given duration."""
    from db import get_all_rooms
    rooms = await get_all_rooms()
    available = []
    for room in rooms:
        is_available = await check_availability(room, booking_date, booking_time, duration_minutes)
        if is_available:
            available.append(room)
    if available:
        return f"Available rooms on {booking_date} at {booking_time}: {', '.join(available)}."
    return f"No rooms are available on {booking_date} at {booking_time}."