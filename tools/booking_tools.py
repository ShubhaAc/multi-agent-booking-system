from langchain_core.tools import tool
from db import (
    check_doctor_availability as db_check_availability,
    create_appointment as db_create_appointment,
    find_alternative_doctor as db_find_alternative,
    get_all_doctors as db_get_all_doctors,
    get_doctor_schedule as db_get_doctor_schedule,
    get_doctors_by_specialization as db_get_doctors_by_specialization,
)

REASON_TO_SPECIALIZATIONS = {
    "toothache / pain":             ["Emergency Dentistry", "Endodontics", "General"],
    "routine checkup & cleaning":   ["General"],
    "filling / cavity":             ["General"],
    "root canal":                   ["Endodontics"],
    "teeth whitening / cosmetic":   ["Cosmetic"],
    "braces / aligners":            ["Orthodontics"],
    "wisdom tooth / extraction":    ["Oral Surgery"],
    "dental implants":              ["Implantology", "Oral Surgery"],
    "child / kid's appointment":    ["Paediatric"],
    "gum problems":                 ["Periodontics"],
    "crowns / bridges / dentures":  ["Prosthodontics"],
    "emergency":                    ["Emergency Dentistry"],
    "other":                        ["General"],
}

@tool
async def check_doctor_availability(doctor_name: str, appointment_date: str, appointment_time: str, duration_minutes: int = 30) -> str:
    """Check if a doctor is available on a given date and time for a given duration."""
    available = await db_check_availability(doctor_name, appointment_date, appointment_time, duration_minutes)
    if available:
        return f"{doctor_name} is available on {appointment_date} at {appointment_time} for {duration_minutes} minutes."
    return f"{doctor_name} is not available on {appointment_date} at {appointment_time} for {duration_minutes} minutes."



@tool
async def book_appointment(doctor_name: str, patient_name: str, patient_email: str, appointment_date: str, appointment_time: str, duration_minutes: int = 30, phone_number: str = None, reason_for_visit: str = None) -> str:
    """Book an appointment with a doctor for a given date, time and duration."""
    appointment_id = await db_create_appointment(
        doctor_name, patient_name, patient_email,
        appointment_date, appointment_time, duration_minutes,
        phone_number=phone_number, reason_for_visit=reason_for_visit
    )
    return f"Appointment confirmed. Appointment ID is {appointment_id}. Duration: {duration_minutes} minutes."



@tool
async def find_alternative_doctor(doctor_name: str, appointment_date: str, appointment_time: str, duration_minutes: int = 30) -> str:
    """Find an alternative available doctor when the requested doctor is unavailable."""
    alternative = await db_find_alternative(doctor_name, appointment_date, appointment_time, duration_minutes)
    if alternative:
        return f"{alternative} is available on {appointment_date} at {appointment_time} for {duration_minutes} minutes."
    return "No alternative doctors are available at that time."



@tool
async def list_available_doctors(appointment_date: str, appointment_time: str, duration_minutes: int = 30) -> str:
    """List all doctors available on a given date and time for a given duration."""
    doctors = await db_get_all_doctors()
    available = []
    for doctor in doctors:
        is_available = await db_check_availability(doctor, appointment_date, appointment_time, duration_minutes)
        if is_available:
            available.append(doctor)
    if available:
        return f"Available doctors on {appointment_date} at {appointment_time}: {', '.join(available)}."
    return f"No doctors are available on {appointment_date} at {appointment_time}."



@tool
async def get_doctor_schedule(doctor_name: str) -> str:
    """Get a doctor's general working days and hours when no specific date/time is given."""
    schedule = await db_get_doctor_schedule(doctor_name)
    if not schedule:
        return f"Could not find a doctor matching '{doctor_name}'."
    return (
        f"{schedule['name']} is available on {schedule['available_days']} "
        f"from {schedule['start_time']} to {schedule['end_time']}."
    )



@tool
async def recommend_doctors(reason_for_visit: str, appointment_date: str = None, appointment_time: str = None, duration_minutes: int = 30) -> str:
    """
    Find and return doctors matching the patient's reason for visit.
    If appointment_date and appointment_time are provided, only returns doctors available at that slot.
    If no date/time provided, returns all matching doctors with their schedules.
    """
    reason_lower = reason_for_visit.lower().strip()

    specializations = None
    for key, specs in REASON_TO_SPECIALIZATIONS.items():
        if any(word in reason_lower for word in key.split(" / ")) or key in reason_lower:
            specializations = specs
            break

    if not specializations:
        specializations = ["General"]

    doctors = await db_get_doctors_by_specialization(specializations)

    if not doctors:
        return "no_matching_doctors"

    if appointment_date and appointment_time:
        available = []
        for doc in doctors:
            if await db_check_availability(doc["name"], appointment_date, appointment_time, duration_minutes):
                available.append(doc)
        if not available:
            return f"no_available_doctors_for_slot|searched={[d['name'] for d in doctors]}"
        return str(available)

    return str(doctors)