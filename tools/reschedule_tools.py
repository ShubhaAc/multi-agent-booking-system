from langchain_core.tools import tool
from db import (
    reschedule_appointment as db_reschedule,
    check_doctor_availability_excluding as db_check_excluding,
    get_appointment_by_id as db_get_appointment_by_id,
)


@tool
async def check_reschedule_availability(
    appointment_id: int,
    doctor_name: str,
    new_date: str,
    new_time: str,
    duration_minutes: int = 30,
) -> str:
    """
    Check if a doctor is available for a new date/time during a reschedule.
    Excludes the appointment being rescheduled from the conflict check so it
    doesn't block itself.
    """
    available = await db_check_excluding(
        doctor_name, new_date, new_time, duration_minutes,
        exclude_appointment_id=appointment_id,
    )
    if available:
        return f"{doctor_name} is available on {new_date} at {new_time} for {duration_minutes} minutes."
    return f"{doctor_name} is not available on {new_date} at {new_time}."


@tool
async def reschedule_appointment(
    appointment_id: int,
    patient_email: str,
    new_doctor_name: str,
    new_date: str,
    new_time: str,
    new_duration_minutes: int = 30,
) -> str:
    """
    Reschedule an existing appointment in-place (UPDATE, not cancel+rebook).
    Verifies ownership, checks availability, and updates the record atomically.
    The appointment ID stays the same.
    """
    result = await db_reschedule(
        appointment_id, patient_email, new_doctor_name,
        new_date, new_time, new_duration_minutes,
    )
    if result["success"]:
        appt = result["appointment"]
        return (
            f"Appointment {appt['id']} has been successfully rescheduled. "
            f"New details: {appt['doctor_name']} on {appt['appointment_date']} "
            f"at {appt['appointment_time']} for {appt['duration_minutes']} minutes."
        )
    if result["reason"] == "not_found":
        return f"No appointment found with ID {appointment_id}."
    if result["reason"] == "not_owner":
        return f"Appointment {appointment_id} does not belong to your account."
    if result["reason"] == "already_cancelled":
        return f"Appointment {appointment_id} is already cancelled and cannot be rescheduled."
    if result["reason"] == "doctor_not_found":
        return f"Could not find a doctor matching '{new_doctor_name}'."
    if result["reason"] == "unavailable":
        return f"{new_doctor_name} is not available on {new_date} at {new_time}."
    return f"Could not reschedule appointment {appointment_id}."