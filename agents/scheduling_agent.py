import logging
import re
from datetime import datetime
from state import GraphState
from db import (
    check_doctor_availability as db_check_availability,
    check_doctor_availability_excluding as db_check_excluding,
    reschedule_appointment as db_reschedule,
    find_alternative_doctor as db_find_alternative,
    get_all_doctors as db_get_all_doctors,
    get_doctor_schedule as db_get_doctor_schedule,
    get_doctors_by_specialization as db_get_doctors_by_specialization,
    get_doctor_specialization as db_get_doctor_specialization,
    get_appointment_by_id as db_get_appointment_by_id,
    get_doctor_day_info as db_get_doctor_day_info,
)
from tools.email_sender import send_reschedule_email

logger = logging.getLogger(__name__)


def _format_time_range(time_str: str, duration_minutes: int) -> str:
    start_h, start_m = int(time_str[:2]), int(time_str[3:])
    total = start_h * 60 + start_m + duration_minutes
    end_h, end_m = divmod(total, 60)
    return f"{time_str}\u2013{end_h:02d}:{end_m:02d}"


def _to_minutes(time_str: str) -> int:
    return int(time_str[:2]) * 60 + int(time_str[3:])


def _to_hhmm(minutes: int) -> str:
    h, m = divmod(minutes, 60)
    return f"{h:02d}:{m:02d}"


def _compute_open_slots(
    start_time: str, end_time: str, duration_minutes: int, bookings: list[dict], step_minutes: int = 30, limit: int = 5,
) -> list[str]:
    
    start_mins = _to_minutes(start_time)
    end_mins = _to_minutes(end_time)
    booked_ranges = [(_to_minutes(b["time"]), _to_minutes(b["time"]) + b["duration"]) for b in bookings]

    slots = []
    t = start_mins
    while t + duration_minutes <= end_mins and len(slots) < limit:
        slot_end = t + duration_minutes
        overlaps = any(t < be and slot_end > bs for bs, be in booked_ranges)
        if not overlaps:
            slots.append(_to_hhmm(t))
        t += step_minutes
    return slots


async def _explain_unavailable_with_specialist(
    exclude_doctor: str,
    appointment_date: str,
    appointment_time: str,
    duration_minutes: int,
    candidate_names: list[str] | None,
) -> str | None:
  
    if not candidate_names:
        return None

    doctor_summaries = []
    for name in candidate_names:
        if name == exclude_doctor:
            continue
        day_info = await db_get_doctor_day_info(name, appointment_date)
        if not day_info or not day_info["works_that_day"]:
            continue

        open_slots = _compute_open_slots(
            day_info["start_time"], day_info["end_time"], duration_minutes, day_info["bookings"],
        )
        if open_slots:
            slots_text = ", ".join(open_slots)
            doctor_summaries.append(f"{day_info['name']} is free at {slots_text} that day")
        else:
            doctor_summaries.append(f"{day_info['name']} works {day_info['weekday']}s but is fully booked that day")

    if not doctor_summaries:
        return None

    return " ".join(doctor_summaries) + ". Would any of those work?"


def _primary_specialization_keyword(full_specialization: str) -> str | None:
    
    if not full_specialization:
        return None
    words = full_specialization.strip().split()
    return words[0] if words else None


async def _resolve_specialization(state: GraphState, fallback_doctor_name: str | None) -> str | None:
   
    if state.specialization_needed:
        return state.specialization_needed
    if fallback_doctor_name:
        full_spec = await db_get_doctor_specialization(fallback_doctor_name)
        return _primary_specialization_keyword(full_spec)
    return None



_ANY_DOCTOR_PATTERN = re.compile(
    r"^\s*(whoever'?s? (is )?free|any(one|body)?( doctor)?|no preference|"
    r"doesn'?t matter|don'?t (know|mind|care)|dont know|dunno|first available)\s*[!.]*\s*$",
    re.IGNORECASE,
)

_DOCTOR_CHANGE_SIGNAL_PATTERN = re.compile(
    r"\b(another|different|someone else|switch|change (my |the )?doctor|new doctor)\b",
    re.IGNORECASE,
)


_AFFIRMATIVE_PATTERN = re.compile(
    r'^\s*(ok(ay)?|yes|yeah|yep|sure|go ahead|sounds good|confirm(ed)?)\s*[!.]*\s*$',
    re.IGNORECASE,
)


async def scheduling_node(state: GraphState) -> dict:
    logger.info("Scheduling agent started.")

    #  RESCHEDULE 
    if state.intent == "reschedule":

        if not state.appointment_id:
            return {"response_message": "Sure, I can help reschedule! Could you share the appointment ID? You'll find it in your confirmation email."}

        if not state.sender_id:
            return {
                "appointment_id": state.appointment_id,
                "response_message": "I need to verify your identity before rescheduling. Could you share the email you used when booking?",
            }


        existing_appt = await db_get_appointment_by_id(state.appointment_id)
        if not existing_appt:
            return {"response_message": f"I couldn't find an appointment with ID {state.appointment_id}. Please double-check the ID."}

        if not state.appointment_date or not state.appointment_time:
            if state.appointment_date and not state.appointment_time:
                msg = f"What time on {state.appointment_date} works for you?"
            elif state.appointment_time and not state.appointment_date:
                msg = f"What date works for {state.appointment_time}?"
            else:
                msg = "What date and time would you like to reschedule to? Also let me know if you'd like to keep the same doctor or switch to someone else."
            return {
                "appointment_id": state.appointment_id,
                "response_message": msg,
            }

        if not state.doctor_name:
            wants_different_doctor = (
                _ANY_DOCTOR_PATTERN.match(state.user_message)
                or _DOCTOR_CHANGE_SIGNAL_PATTERN.search(state.user_message)
            )
            if wants_different_doctor:

                specialization_needed = await _resolve_specialization(state, existing_appt["doctor_name"])
                candidate_names = None
                if specialization_needed:
                    same_specialty = await db_get_doctors_by_specialization([specialization_needed])
                    candidate_names = [d["name"] for d in same_specialty]

                pool = candidate_names if candidate_names else await db_get_all_doctors()
              
                pool = [name for name in pool if name != existing_appt["doctor_name"]]

                available_candidates = []
                for name in pool:
                    if await db_check_excluding(
                        name, state.appointment_date, state.appointment_time,
                        state.duration_minutes, exclude_appointment_id=state.appointment_id,
                    ):
                        available_candidates.append(name)

                if not available_candidates:
                    explanation = await _explain_unavailable_with_specialist(
                        existing_appt["doctor_name"], state.appointment_date, state.appointment_time,
                        state.duration_minutes, candidate_names,
                    )
                    base = (
                        f"No other doctors are free on {state.appointment_date} at {state.appointment_time}."
                    )
                    return {
                        "appointment_id": state.appointment_id,
                        "appointment_date": state.appointment_date,
                        "appointment_time": state.appointment_time,
                        "response_message": (
                            f"{base} {explanation}" if explanation
                            else f"{base} Could you try a different date or time?"
                        ),
                    }

                if len(available_candidates) > 1:
                    names_text = ", ".join(available_candidates)
                    return {
                        "appointment_id": state.appointment_id,
                        "appointment_date": state.appointment_date,
                        "appointment_time": state.appointment_time,
                        "response_message": (
                            f"{names_text} are all free on {state.appointment_date} at "
                            f"{state.appointment_time} — who would you prefer?"
                        ),
                    }

                chosen = available_candidates[0]
                return {
                    "appointment_id": state.appointment_id,
                    "appointment_date": state.appointment_date,
                    "appointment_time": state.appointment_time,
                    "response_message": (
                        f"{chosen} is free on {state.appointment_date} at {state.appointment_time} "
                        f"— shall I go ahead and reschedule you to them?"
                    ),
                    "suggested_alternative": chosen,
                }

            return {
                "appointment_id": state.appointment_id,
                "appointment_date": state.appointment_date,
                "appointment_time": state.appointment_time,
                "response_message": "Which doctor would you like to reschedule with? I can keep the same one or find someone new if you prefer.",
            }


        if (
            state.appointment_date == existing_appt["appointment_date"]
            and state.appointment_time == existing_appt["appointment_time"]
            and state.doctor_name == existing_appt["doctor_name"]
        ):
            return {
                "appointment_id": state.appointment_id,
                "response_message": (
                    f"Appointment {state.appointment_id} is currently set for "
                    f"{existing_appt['appointment_date']} at {existing_appt['appointment_time']} "
                    f"with {existing_appt['doctor_name']}. What new date, time, or doctor would you "
                    f"like to change it to?"
                ),
            }

        specialization_needed = await _resolve_specialization(state, existing_appt["doctor_name"])


        schedule = await db_get_doctor_schedule(state.doctor_name)
        if schedule:
            weekday = datetime.strptime(state.appointment_date, "%Y-%m-%d").strftime("%A")
            working_days = [d.strip() for d in schedule["available_days"].split(",")]
            if weekday not in working_days:
                candidate_names = None
                if specialization_needed:
                    same_specialty = await db_get_doctors_by_specialization([specialization_needed])
                    candidate_names = [d["name"] for d in same_specialty]

                alternative = await db_find_alternative(
                    state.doctor_name,
                    state.appointment_date,
                    state.appointment_time,
                    state.duration_minutes,
                    candidate_names=candidate_names,
                )
                if alternative:
                    return {
                        "appointment_id": state.appointment_id,
                        "response_message": (
                            f"{schedule['name']} is only available {schedule['available_days']} and "
                            f"doesn't work on {weekday}s. {alternative} is free on {state.appointment_date} "
                            f"at {state.appointment_time} instead — would you like to reschedule with them?"
                        ),
                        "suggested_alternative": alternative,
                    }

                explanation = await _explain_unavailable_with_specialist(
                    state.doctor_name, state.appointment_date, state.appointment_time,
                    state.duration_minutes, candidate_names,
                )
                base = (
                    f"{schedule['name']} is only available {schedule['available_days']} and doesn't "
                    f"work on {weekday}s."
                )
                if explanation:
                    return {
                        "appointment_id": state.appointment_id,
                        "response_message": f"{base} {explanation}",
                    }
                return {
                    "appointment_id": state.appointment_id,
                    "response_message": f"{base} Would you like to try one of those days instead?",
                }

        # Check availability excluding this appointment
        available = await db_check_excluding(
            state.doctor_name,
            state.appointment_date,
            state.appointment_time,
            state.duration_minutes,
            exclude_appointment_id=state.appointment_id,
        )

        if not available:
            # Restrict alternatives to the same specialty as the original
      
            candidate_names = None
            if specialization_needed:
                same_specialty = await db_get_doctors_by_specialization([specialization_needed])
                candidate_names = [d["name"] for d in same_specialty]

            alternative = await db_find_alternative(
                state.doctor_name,
                state.appointment_date,
                state.appointment_time,
                state.duration_minutes,
                candidate_names=candidate_names,
            )
            if alternative:
                return {
                    "appointment_id": state.appointment_id,
                    "response_message": (
                        f"{state.doctor_name} isn't available on {state.appointment_date} at {state.appointment_time}. "
                        f"However, {alternative} is free at that time — would you like to reschedule with them instead?"
                    ),
                    "suggested_alternative": alternative,
                }

            explanation = await _explain_unavailable_with_specialist(
                state.doctor_name, state.appointment_date, state.appointment_time,
                state.duration_minutes, candidate_names,
            )
            base = f"{state.doctor_name} isn't available on {state.appointment_date} at {state.appointment_time}."
            if explanation:
                return {
                    "appointment_id": state.appointment_id,
                    "response_message": f"{base} {explanation}",
                }
            return {
                "appointment_id": state.appointment_id,
                "response_message": f"{base} No other doctors are free then either. Could you try a different date or time?",
            }

        # CONFIRM BEFORE COMMITTING 
        if not _AFFIRMATIVE_PATTERN.match(state.user_message):
            return {
                "appointment_id": state.appointment_id,
                "appointment_date": state.appointment_date,
                "appointment_time": state.appointment_time,
                "response_message": (
                    f"{state.doctor_name} is free on {state.appointment_date} at "
                    f"{state.appointment_time} — shall I go ahead and reschedule you?"
                ),
                "suggested_alternative": state.doctor_name,
            }

        # Reschedule directly using sender_id for ownership
        result = await db_reschedule(
            appointment_id=state.appointment_id,
            sender_id=state.sender_id,
            new_doctor_name=state.doctor_name,
            new_date=state.appointment_date,
            new_time=state.appointment_time,
            new_duration_minutes=state.duration_minutes,
        )

        if not result["success"]:
            if result["reason"] == "not_owner":
                return {"response_message": f"Appointment {state.appointment_id} doesn't appear to belong to your account."}
            if result["reason"] == "already_cancelled":
                return {"response_message": f"Appointment {state.appointment_id} is already cancelled and can't be rescheduled."}
            if result["reason"] == "doctor_not_found":
                return {"response_message": f"I couldn't find a doctor matching '{state.doctor_name}'."}
            if result["reason"] == "unavailable":
                return {"response_message": f"{state.doctor_name} is not available on {state.appointment_date} at {state.appointment_time}."}
            return {"response_message": f"Could not reschedule appointment {state.appointment_id}. Please try again."}

        appt = result["appointment"]
        logger.info("Appointment %s rescheduled.", state.appointment_id)

        try:
            await send_reschedule_email.ainvoke({
                "to_email": state.invitee_email,
                "doctor_name": appt["doctor_name"],
                "new_date": appt["appointment_date"],
                "new_time": appt["appointment_time"],
                "appointment_id": appt["id"],
            })
            logger.info("Reschedule email sent to %s", state.invitee_email)
            email_note = f" A confirmation email has been sent to {state.invitee_email}."
        except Exception as e:
            logger.error("Failed to send reschedule email: %s", e)
            email_note = ""

        return {
            "response_message": (
                f"Done! Your appointment (ID: {appt['id']}) has been rescheduled to "
                f"{appt['appointment_date']} at {appt['appointment_time']} with {appt['doctor_name']}."
                f"{email_note}"
            ),
            "appointment_id": appt["id"],
            "suggested_alternative": None,
            "intent": None,
        }

    #  CHECK AVAILABILITY 
    elif state.intent == "check_availability":

        if not state.doctor_name:
            if not state.appointment_date:
                return {"response_message": "Please specify a date to check doctor availability."}

            if state.specialization_needed:
                doctors = await db_get_doctors_by_specialization([state.specialization_needed])
                doctor_names = [d["name"] for d in doctors]
            else:
                doctor_names = await db_get_all_doctors()

            weekday = datetime.strptime(state.appointment_date, "%Y-%m-%d").strftime("%A")

            if not state.appointment_time:
                free_that_day = []
                for doctor in doctor_names:
                    schedule = await db_get_doctor_schedule(doctor)
                    if schedule and weekday in [d.strip() for d in schedule["available_days"].split(",")]:
                        free_that_day.append(f"{doctor} ({schedule['start_time']}–{schedule['end_time']})")

                if free_that_day:
                    return {"response_message": f"On {weekday}s, the following doctors are available: {', '.join(free_that_day)}."}
                return {"response_message": f"No doctors work on {weekday}s. Could you try a different day?"}

            available = []
            for doctor in doctor_names:
                if await db_check_availability(doctor, state.appointment_date, state.appointment_time, state.duration_minutes):
                    available.append(doctor)

            if available:
                return {"response_message": f"The following doctors are available on {state.appointment_date} at {state.appointment_time}: {', '.join(available)}."}
            return {"response_message": f"No doctors are available on {state.appointment_date} at {state.appointment_time} for {state.specialization_needed or 'your reason'}."}

        schedule = await db_get_doctor_schedule(state.doctor_name)
        if not schedule:
            return {"response_message": f"I couldn't find a doctor matching '{state.doctor_name}'."}

        if not state.appointment_date and not state.appointment_time:
            return {"response_message": f"{schedule['name']} is available on {schedule['available_days']} from {schedule['start_time']} to {schedule['end_time']}."}

        if state.appointment_date and not state.appointment_time:
            day_of_week = datetime.strptime(state.appointment_date, "%Y-%m-%d").strftime("%A")
            if day_of_week in schedule["available_days"].split(","):
                return {"response_message": f"Yes, {schedule['name']} is available on {state.appointment_date} from {schedule['start_time']} to {schedule['end_time']}."}
            return {"response_message": f"{schedule['name']} does not work on {state.appointment_date} ({day_of_week}s)."}

        if not state.appointment_date and state.appointment_time:
            return {"response_message": f"Please specify a date to check availability for {state.doctor_name} at {state.appointment_time}."}

        available = await db_check_availability(
            state.doctor_name, state.appointment_date, state.appointment_time, state.duration_minutes
        )
        if available:
            return {"response_message": f"{state.doctor_name} is available on {state.appointment_date} at {state.appointment_time}."}
        return {"response_message": f"{state.doctor_name} is not available on {state.appointment_date} at {state.appointment_time}."}

    #  FALLBACK 
    else:
        if not state.doctor_name or not state.appointment_date or not state.appointment_time:
            return {"response_message": "Please specify the doctor, date, and time so I can find an alternative."}
        alternative = await db_find_alternative(
            state.doctor_name, state.appointment_date, state.appointment_time, state.duration_minutes
        )
        if alternative:
            return {
                "response_message": f"{state.doctor_name} is unavailable at that time. {alternative} is available instead — would you like to book with them?",
                "suggested_alternative": alternative,
            }
        return {"response_message": f"No doctors are available on {state.appointment_date} at {state.appointment_time}. Could you try a different time?"}