from langchain_core.tools import tool
from db import (
    cancel_appointment_with_ownership as db_cancel_with_ownership,
    get_appointment_by_id as db_get_appointment_by_id,
)


@tool
async def get_appointment_details(appointment_id: int) -> str:
    """Look up an appointment's doctor, date, time, and patient email by its ID."""
    appt = await db_get_appointment_by_id(appointment_id)
    if not appt:
        return f"No appointment found with ID {appointment_id}."
    return (
        f"Appointment {appt['id']}: doctor_name={appt['doctor_name']}, "
        f"appointment_date={appt['appointment_date']}, appointment_time={appt['appointment_time']}, "
        f"patient_email={appt['patient_email']}, status={appt['status']}"
    )


@tool
async def cancel_appointment(appointment_id: int, patient_email: str) -> str:
    """
    Cancel an active appointment by its ID.
    Only succeeds if the appointment belongs to patient_email.
    """
    result = await db_cancel_with_ownership(appointment_id, patient_email)
    if result["success"]:
        return f"Appointment {appointment_id} has been successfully cancelled."
    if result["reason"] == "not_found":
        return f"No appointment found with ID {appointment_id}."
    if result["reason"] == "not_owner":
        return f"Appointment {appointment_id} does not belong to your account and cannot be cancelled."
    if result["reason"] == "already_cancelled":
        return f"Appointment {appointment_id} is already cancelled."
    return f"Could not cancel appointment {appointment_id}."