import logging
import re
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from state import GraphState
from db import (
    check_doctor_availability as db_check_availability,
    create_appointment as db_create_appointment,
    find_alternative_doctor as db_find_alternative,
    get_doctors_by_specialization as db_get_doctors_by_specialization,
)
from tools.email_sender import send_invite_email
from agents.scheduling_agent import _explain_unavailable_with_specialist
from config import MODEL_NAME



logger = logging.getLogger(__name__)
llm = ChatOpenAI(model=MODEL_NAME, temperature=0)


_AFFIRMATIVE_PATTERN = re.compile(
    r'^\s*(ok(ay)?|yes|yeah|yep|sure|go ahead|sounds good|confirm(ed)?)\s*[!.]*\s*$',
    re.IGNORECASE,
)

REASON_MENU = (
    "1. Toothache / Pain\n"
    "2. Routine Checkup & Cleaning\n"
    "3. Filling / Cavity\n"
    "4. Root Canal\n"
    "5. Teeth Whitening / Cosmetic\n"
    "6. Braces / Aligners\n"
    "7. Wisdom Tooth / Extraction\n"
    "8. Dental Implants\n"
    "9. Child / Kid's Appointment\n"
    "10. Gum Problems\n"
    "11. Crowns / Bridges / Dentures\n"
    "12. Emergency\n"
    "13. Other (describe briefly)"
    
)

REASON_TO_SPECIALIZATIONS = {
    "toothache / pain":            ["Emergency Dentistry", "Endodontics", "General"],
    "routine checkup & cleaning":  ["General"],
    "filling / cavity":            ["General"],
    "root canal":                  ["Endodontics"],
    "teeth whitening / cosmetic":  ["Cosmetic"],
    "braces / aligners":           ["Orthodontics"],
    "wisdom tooth / extraction":   ["Oral Surgery"],
    "dental implants":             ["Implantology", "Oral Surgery"],
    "child / kid's appointment":   ["Paediatric"],
    "gum problems":                ["Periodontics"],
    "crowns / bridges / dentures": ["Prosthodontics"],
    "emergency":                   ["Emergency Dentistry"],
    "other":                       ["General"],
}


async def _get_doctors_for_reason(reason: str) -> list[dict]:
    reason_lower = reason.lower().strip()
    specializations = None
    for key, specs in REASON_TO_SPECIALIZATIONS.items():
        if any(word in reason_lower for word in key.split(" / ")) or key in reason_lower:
            specializations = specs
            break
    if not specializations:
        specializations = ["General"]
    return await db_get_doctors_by_specialization(specializations)


def _doctor_works_on_date(doctor: dict, appointment_date: str) -> bool:
    try:
        weekday = datetime.strptime(appointment_date, "%Y-%m-%d").strftime("%A")
    except (ValueError, TypeError):
        return True
    days = [d.strip() for d in doctor.get("available_days", "").split(",")]
    return weekday in days


async def _recommend_doctors_response(state: GraphState) -> str:
    doctors = await _get_doctors_for_reason(state.reason_for_visit)

    if state.appointment_date and state.appointment_time:
        available = []
        for doc in doctors:
            if await db_check_availability(doc["name"], state.appointment_date, state.appointment_time, state.duration_minutes):
                available.append(doc)
        doctors_to_show = available
        has_slot = True
    else:
        doctors_to_show = doctors
        has_slot = False

    if not doctors_to_show:
        if has_slot:
            names = ", ".join(d["name"] for d in doctors)
            context = f"No doctors available at {state.appointment_date} {state.appointment_time}. Normally handle this: {names}."
        else:
            context = "No matching specialists found."
    else:
        context = "\n".join(
            f"- {d['name']} ({d['specialization']}) — available {d['available_days']} {d['start_time']}–{d['end_time']}"
            for d in doctors_to_show
        )

    doctor_count = len(doctors_to_show)
    if doctor_count > 1:
        mention_instruction = (
            f"There are {doctor_count} matching doctors — mention EVERY one of them by name "
            f"(do not omit any), with a brief note on what distinguishes each, so the patient can choose."
        )
    else:
        mention_instruction = "Mention the doctor by name and briefly say why they're a good fit."

    prompt = (
        f"You are a warm dental clinic receptionist. The patient's reason: '{state.reason_for_visit}'.\n"
        f"Matching doctors:\n{context}\n\n"
        f"{'Specific slot requested: ' + state.appointment_date + ' at ' + state.appointment_time + '.' if has_slot else 'No date/time given yet.'}\n\n"
        f"Respond naturally in 2-4 sentences. {mention_instruction} "
        f"{'Confirm who is free at that slot and ask the patient to pick if more than one.' if has_slot else 'End by asking when works best for the patient.'} "
        f"No bullet points, no numbering, no jargon."
    )

    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return response.content.strip()


async def booking_node(state: GraphState) -> dict:
    logger.info("Booking agent started.")

    # Phase 1: name + phone
    missing_contact = []
    if not state.patient_name:
        missing_contact.append("your full name")
    if not state.phone_number:
        missing_contact.append("your phone number")
    if missing_contact:
        logger.info("Booking request missing contact info: %s", missing_contact)
        return {"response_message": f"Before I proceed, could I get {' and '.join(missing_contact)}?"}

    # Phase 2: email + reason
    missing_email = not state.invitee_email
    missing_reason = not state.reason_for_visit
    if missing_email or missing_reason:
        logger.info("Booking request missing email/reason: email=%s reason=%s", missing_email, missing_reason)
        if missing_email and missing_reason:
            return {"response_message": f"Thanks! Could you also share your email address, and what brings you in?\n\n{REASON_MENU}"}
        if missing_reason:
            return {"response_message": f"Thanks! And what brings you in?\n\n{REASON_MENU}"}
        return {"response_message": "Thanks! Could you also share your email address?"}

    # Phase 3: no doctor yet — recommend
    if not state.doctor_name:
        candidates = await _get_doctors_for_reason(state.reason_for_visit)

        if not candidates:
            return {"response_message": "I couldn't find a specialist matching that reason — could you tell me a bit more about what's going on?"}


        # date-aware selection 
        if not state.appointment_date:
            response = await _recommend_doctors_response(state)
            logger.info("Booking recommendation response: %s", response)
            return {"response_message": response, "recommended_doctors": [d["name"] for d in candidates]}

        weekday = datetime.strptime(state.appointment_date, "%Y-%m-%d").strftime("%A")
        day_matches = [
            d for d in candidates
            if weekday in [day.strip() for day in d.get("available_days", "").split(",")]
        ]

        if not day_matches:
            names = ", ".join(d["name"] for d in candidates)
            if len(candidates) == 1:
                return {"response_message": f"{candidates[0]['name']} doesn't work on {weekday}s. Could you try a different day?"}
            return {"response_message": f"None of the doctors for this reason ({names}) work on {weekday}s. Could you try a different day?"}

        if not state.appointment_time:
            names = ", ".join(f"{d['name']} ({d['start_time']}–{d['end_time']})" for d in day_matches)
            return {
                "response_message": f"On {weekday}s, {names} {'is' if len(day_matches) == 1 else 'are'} available. What time works for you?",
                "recommended_doctors": [d["name"] for d in day_matches],
            }


        available = []
        for d in day_matches:
            if await db_check_availability(d["name"], state.appointment_date, state.appointment_time, state.duration_minutes):
                available.append(d)

        if not available:
            names = ", ".join(d["name"] for d in day_matches)
            if len(day_matches) == 1:
                return {"response_message": f"{day_matches[0]['name']} isn't free at {state.appointment_time} on {state.appointment_date}. Could you try a different time?"}
            return {"response_message": f"None of {names} are free at {state.appointment_time} on {state.appointment_date}. Could you try a different time?"}

        if len(available) > 1:
            names = ", ".join(d["name"] for d in available)
            return {
                "response_message": f"Both {names} are free at {state.appointment_time} on {state.appointment_date} — who would you prefer?",
                "recommended_doctors": [d["name"] for d in available],
            }

        d = available[0]
        return {
            "response_message": f"{d['name']} is free on {state.appointment_date} at {state.appointment_time} — shall I go ahead and book that for you?",
            "suggested_alternative": d["name"],
            "recommended_doctors": [d["name"]],
        }

    # Phase 4: doctor known, missing date/time
    missing_booking = []
    if not state.appointment_date:
        missing_booking.append("appointment date")
    if not state.appointment_time:
        missing_booking.append("appointment time")
    if missing_booking:
        logger.info("Booking request missing scheduling fields: %s", missing_booking)
        return {"response_message": f"Got it! Could you also share the {' and '.join(missing_booking)} for the appointment with {state.doctor_name}?"}

    # Final guard
    if not (state.patient_name and state.phone_number and state.invitee_email and state.reason_for_visit):
        return {"response_message": "I still need your name, phone number, email, and reason for visit — could you share whichever is missing?"}

    # Phase 5: direct DB calls
    logger.info("Booking phase 5: checking availability for %s on %s at %s", state.doctor_name, state.appointment_date, state.appointment_time)

    available = await db_check_availability(
        state.doctor_name, state.appointment_date, state.appointment_time, state.duration_minutes
    )

    if not available:
        candidates = await _get_doctors_for_reason(state.reason_for_visit)
        candidate_names = [d["name"] for d in candidates]
        alternative = await db_find_alternative(
            state.doctor_name,
            state.appointment_date,
            state.appointment_time,
            state.duration_minutes,
            candidate_names=candidate_names,
        )
        if alternative:
            return {
                "response_message": (
                    f"{state.doctor_name} isn't available on {state.appointment_date} at {state.appointment_time}. "
                    f"However, {alternative} is free at that time — would you like to book with them instead?"
                ),
                "suggested_alternative": alternative,
            }
        explanation = await _explain_unavailable_with_specialist(
            state.doctor_name, state.appointment_date, state.appointment_time,
            state.duration_minutes, candidate_names,
        )
        base = f"Unfortunately {state.doctor_name} isn't available at that time, and no other doctors matching your needs are free then either."
        if explanation:
            return {"response_message": f"{base} {explanation}"}
        return {"response_message": f"{base} Could you try a different date or time?"}


    if not _AFFIRMATIVE_PATTERN.match(state.user_message):
        return {
            "response_message": (
                f"{state.doctor_name} is free on {state.appointment_date} at "
                f"{state.appointment_time} — shall I go ahead and book that for you?"
            ),
            "suggested_alternative": state.doctor_name,
        }

    appointment_id = await db_create_appointment(
        doctor_name=state.doctor_name,
        patient_name=state.patient_name,
        patient_email=state.invitee_email,
        appointment_date=state.appointment_date,
        appointment_time=state.appointment_time,
        duration_minutes=state.duration_minutes,
        phone_number=state.phone_number,
        reason_for_visit=state.reason_for_visit,
        sender_id=state.sender_id,
    )
    logger.info("Appointment created with id %s", appointment_id)

    email_result = await send_invite_email.ainvoke({
        "to_email": state.invitee_email,
        "doctor_name": state.doctor_name,
        "booking_date": state.appointment_date,
        "booking_time": state.appointment_time,
        "appointment_id": appointment_id,
    })
    logger.info("Email result: %s", email_result)

    return {
            "response_message": (
                f"You're all set! Your appointment with {state.doctor_name} is confirmed for "
                f"{state.appointment_date} at {state.appointment_time} (appointment ID: {appointment_id}). "
                f"A confirmation email has been sent to {state.invitee_email}."
            ),
            "appointment_id": appointment_id,
            "suggested_alternative": None,
            "intent": None,
            "doctor_name": None,
            "appointment_date": None,
            "appointment_time": None,
            "reason_for_visit": None,
            "specialization_needed": None,
        }