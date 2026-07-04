import aiosqlite
import logging
from datetime import datetime
from config import DB_PATH

logger = logging.getLogger(__name__)


def _is_valid_time_format(value: str) -> bool:
    if not isinstance(value, str):
        return False
    try:
        datetime.strptime(value, "%H:%M")
        return True
    except ValueError:
        return False


def _is_valid_date_format(value: str) -> bool:
    if not isinstance(value, str):
        return False
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return True
    except ValueError:
        return False


async def _resolve_doctor_name(db, doctor_name: str) -> str | None:
    async with db.execute("SELECT name FROM doctors WHERE name = ?", (doctor_name,)) as cursor:
        exact = await cursor.fetchone()
    if exact:
        return exact[0]
    last_word = doctor_name.strip().split()[-1]
    async with db.execute("SELECT name FROM doctors WHERE name LIKE ?", (f"%{last_word}",)) as cursor:
        matches = await cursor.fetchall()
    if len(matches) == 1:
        return matches[0][0]
    async with db.execute("SELECT name FROM doctors WHERE name LIKE ?", (f"%{doctor_name}%",)) as cursor:
        matches = await cursor.fetchall()
    if len(matches) == 1:
        return matches[0][0]
    return None


async def check_doctor_availability(
    doctor_name: str,
    appointment_date: str,
    appointment_time: str,
    duration_minutes: int = 30,
) -> bool:
    return await check_doctor_availability_excluding(
        doctor_name, appointment_date, appointment_time, duration_minutes,
        exclude_appointment_id=None,
    )


async def check_doctor_availability_excluding(
    doctor_name: str,
    appointment_date: str,
    appointment_time: str,
    duration_minutes: int = 30,
    exclude_appointment_id: int = None,
) -> bool:
    if not _is_valid_date_format(appointment_date):
        logger.warning("Invalid appointment_date received: %r", appointment_date)
        return False
    if not _is_valid_time_format(appointment_time):
        logger.warning("Invalid appointment_time received: %r", appointment_time)
        return False

    async with aiosqlite.connect(DB_PATH) as db:
        resolved_name = await _resolve_doctor_name(db, doctor_name)
        if not resolved_name:
            logger.warning("Doctor not found or ambiguous: %r", doctor_name)
            return False

        async with db.execute(
            "SELECT available_days, start_time, end_time FROM doctors WHERE name = ?",
            (resolved_name,)
        ) as cursor:
            doctor = await cursor.fetchone()
        available_days, start_time, end_time = doctor

        day_of_week = datetime.strptime(appointment_date, "%Y-%m-%d").strftime("%A")
        if day_of_week not in available_days.split(","):
            return False

        appt_mins = int(appointment_time[:2]) * 60 + int(appointment_time[3:])
        start_mins = int(start_time[:2]) * 60 + int(start_time[3:])
        end_mins = int(end_time[:2]) * 60 + int(end_time[3:])
        if appt_mins < start_mins or (appt_mins + duration_minutes) > end_mins:
            return False

        async with db.execute("""
            SELECT a.id FROM appointments a
            JOIN doctors d ON a.doctor_id = d.id
            WHERE d.name = ?
            AND a.appointment_date = ?
            AND a.status = 'active'
            AND a.id != COALESCE(?, -1)
            AND (
                (substr(a.appointment_time, 1, 2) * 60 + substr(a.appointment_time, 4, 2))
                    < (substr(?, 1, 2) * 60 + substr(?, 4, 2)) + ?
                AND
                (substr(a.appointment_time, 1, 2) * 60 + substr(a.appointment_time, 4, 2))
                    + a.duration_minutes
                    > (substr(?, 1, 2) * 60 + substr(?, 4, 2))
            )
        """, (
            resolved_name, appointment_date,
            exclude_appointment_id,
            appointment_time, appointment_time, duration_minutes,
            appointment_time, appointment_time,
        )) as cursor:
            return await cursor.fetchone() is None


async def create_appointment(
    doctor_name: str,
    patient_name: str,
    patient_email: str,
    appointment_date: str,
    appointment_time: str,
    duration_minutes: int = 30,
    phone_number: str = None,
    reason_for_visit: str = None,
    sender_id: str = None,
) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        resolved_name = await _resolve_doctor_name(db, doctor_name)
        if not resolved_name:
            raise ValueError(f"Doctor not found or ambiguous: {doctor_name}")
        async with db.execute("SELECT id FROM doctors WHERE name = ?", (resolved_name,)) as cursor:
            doctor = await cursor.fetchone()
        result = await db.execute("""
            INSERT INTO appointments
                (doctor_id, patient_name, patient_email, phone_number,
                 reason_for_visit, sender_id, appointment_date, appointment_time, duration_minutes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            doctor[0], patient_name, patient_email, phone_number,
            reason_for_visit, sender_id, appointment_date, appointment_time, duration_minutes,
        ))
        await db.commit()
        logger.info("Appointment created with id %s for sender %s", result.lastrowid, sender_id)
        return result.lastrowid


async def cancel_appointment(appointment_id: int) -> bool:
    """Cancel by ID only — no ownership check. Internal use only."""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "UPDATE appointments SET status = 'cancelled' WHERE id = ? AND status = 'active'",
            (appointment_id,)
        )
        await db.commit()
        if cursor.rowcount == 0:
            logger.warning("Cancel failed for appointment %s.", appointment_id)
            return False
        logger.info("Appointment %s cancelled.", appointment_id)
        return True


async def cancel_appointment_with_ownership(
    appointment_id: int,
    sender_id: str,
) -> dict:
    """
    Cancel only if the appointment belongs to this sender_id.
    Returns dict: success bool + reason string.
    Falls back to email match if sender_id was not stored (legacy rows).
    """
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT status, sender_id, patient_email FROM appointments WHERE id = ?",
            (appointment_id,)
        ) as cursor:
            row = await cursor.fetchone()

        if not row:
            logger.warning("Cancel failed for appointment %s: not found.", appointment_id)
            return {"success": False, "reason": "not_found"}

        status, owner_sender_id, owner_email = row

        # Primary check: sender_id match
        if owner_sender_id and owner_sender_id != sender_id:
            logger.warning(
                "Cancel rejected for appointment %s: sender %r is not owner %r.",
                appointment_id, sender_id, owner_sender_id,
            )
            return {"success": False, "reason": "not_owner"}

        if status != "active":
            return {"success": False, "reason": "already_cancelled"}

        await db.execute(
            "UPDATE appointments SET status = 'cancelled' WHERE id = ?",
            (appointment_id,)
        )
        await db.commit()
        logger.info("Appointment %s cancelled by sender %r.", appointment_id, sender_id)
        return {"success": True, "reason": "cancelled"}


async def reschedule_appointment(
    appointment_id: int,
    sender_id: str,
    new_doctor_name: str,
    new_date: str,
    new_time: str,
    new_duration_minutes: int = 30,
) -> dict:
    """
    Reschedule in-place (UPDATE). Verifies sender_id ownership.
    Returns dict with success bool, reason, and updated appointment on success.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT status, sender_id FROM appointments WHERE id = ?",
            (appointment_id,)
        ) as cursor:
            row = await cursor.fetchone()

        if not row:
            return {"success": False, "reason": "not_found"}

        status, owner_sender_id = row

        if owner_sender_id and owner_sender_id != sender_id:
            logger.warning(
                "Reschedule rejected for appointment %s: sender %r is not owner %r.",
                appointment_id, sender_id, owner_sender_id,
            )
            return {"success": False, "reason": "not_owner"}

        if status != "active":
            return {"success": False, "reason": "already_cancelled"}

        resolved_name = await _resolve_doctor_name(db, new_doctor_name)
        if not resolved_name:
            return {"success": False, "reason": "doctor_not_found"}

        async with db.execute("SELECT id FROM doctors WHERE name = ?", (resolved_name,)) as cursor:
            doctor_row = await cursor.fetchone()
        new_doctor_id = doctor_row[0]

    # Availability check excluding this appointment
    available = await check_doctor_availability_excluding(
        resolved_name, new_date, new_time, new_duration_minutes,
        exclude_appointment_id=appointment_id,
    )
    if not available:
        return {"success": False, "reason": "unavailable"}

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            UPDATE appointments
            SET doctor_id = ?, appointment_date = ?, appointment_time = ?, duration_minutes = ?
            WHERE id = ?
        """, (new_doctor_id, new_date, new_time, new_duration_minutes, appointment_id))
        await db.commit()
        logger.info(
            "Appointment %s rescheduled to %s %s with %s by sender %s.",
            appointment_id, new_date, new_time, resolved_name, sender_id,
        )

        async with db.execute("""
            SELECT a.id, a.appointment_date, a.appointment_time, a.duration_minutes,
                   a.status, a.patient_name, a.patient_email, d.name AS doctor_name
            FROM appointments a
            JOIN doctors d ON a.doctor_id = d.id
            WHERE a.id = ?
        """, (appointment_id,)) as cursor:
            updated = await cursor.fetchone()

    return {
        "success": True,
        "reason": "rescheduled",
        "appointment": {
            "id": updated[0],
            "appointment_date": updated[1],
            "appointment_time": updated[2],
            "duration_minutes": updated[3],
            "status": updated[4],
            "patient_name": updated[5],
            "patient_email": updated[6],
            "doctor_name": updated[7],
        },
    }


async def find_alternative_doctor(
    exclude_doctor: str,
    appointment_date: str,
    appointment_time: str,
    duration_minutes: int = 30,
    candidate_names: list[str] = None,
) -> str | None:
    """
    Find a free doctor at the given slot, excluding `exclude_doctor`.
    If `candidate_names` is provided, ONLY those doctors are considered — this
    should be the same specialization-matched candidate list already used to
    make the original recommendation (see booking_agent._get_doctors_for_reason),
    so the alternative offered is guaranteed to be clinically appropriate rather
    than "any doctor who happens to be free" (which previously surfaced e.g. a
    Paediatric Dentist as an "alternative" for an adult general visit).
    If `candidate_names` is None, falls back to searching all doctors (used for
    reschedule flows where the original reason/specialization isn't known).
    """
    async with aiosqlite.connect(DB_PATH) as db:
        resolved_exclude = await _resolve_doctor_name(db, exclude_doctor)
        exclude_value = resolved_exclude or exclude_doctor

        if candidate_names is not None:
            doctors = [name for name in candidate_names if name != exclude_value]
            if not doctors:
                logger.info(
                    "No same-specialization candidates left (excluding %r); "
                    "returning no alternative instead of an unrelated specialist.",
                    exclude_value,
                )
                return None
        else:
            async with db.execute(
                "SELECT name FROM doctors WHERE name != ?",
                (exclude_value,)
            ) as cursor:
                doctors = [row[0] for row in await cursor.fetchall()]

    for doctor in doctors:
        if await check_doctor_availability(doctor, appointment_date, appointment_time, duration_minutes):
            return doctor
    return None


async def get_all_doctors() -> list[str]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT name FROM doctors") as cursor:
            return [row[0] for row in await cursor.fetchall()]


async def get_doctor_schedule(doctor_name: str) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        resolved_name = await _resolve_doctor_name(db, doctor_name)
        if not resolved_name:
            return None
        async with db.execute(
            "SELECT name, available_days, start_time, end_time FROM doctors WHERE name = ?",
            (resolved_name,)
        ) as cursor:
            row = await cursor.fetchone()
        if not row:
            return None
        return {
            "name": row[0],
            "available_days": row[1],
            "start_time": row[2],
            "end_time": row[3],
        }


async def get_doctor_specialization(doctor_name: str) -> str | None:
    """
    Look up a doctor's specialization directly, so callers that only have a
    doctor name (e.g. an in-progress reschedule where the patient never
    restated their reason for visit) can still build a same-specialty
    candidate list instead of silently falling back to "search every doctor",
    which is what caused alternative-doctor suggestions to skip the
    explanatory messaging (see get_doctor_day_info / _explain_unavailable_
    with_specialist in scheduling_agent.py — both require a non-empty
    candidate_names list to run).
    """
    async with aiosqlite.connect(DB_PATH) as db:
        resolved_name = await _resolve_doctor_name(db, doctor_name)
        if not resolved_name:
            return None
        async with db.execute(
            "SELECT specialization FROM doctors WHERE name = ?", (resolved_name,)
        ) as cursor:
            row = await cursor.fetchone()
        return row[0] if row else None


async def get_doctors_by_specialization(specializations: list[str]) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        results = []
        seen = set()
        for keyword in specializations:
            async with db.execute(
                "SELECT name, specialization, available_days, start_time, end_time "
                "FROM doctors WHERE LOWER(specialization) LIKE LOWER(?)",
                (f"%{keyword}%",)
            ) as cursor:
                rows = await cursor.fetchall()
            for row in rows:
                if row[0] not in seen:
                    seen.add(row[0])
                    results.append({
                        "name": row[0],
                        "specialization": row[1],
                        "available_days": row[2],
                        "start_time": row[3],
                        "end_time": row[4],
                    })
        return results


async def get_doctor_day_info(doctor_name: str, appointment_date: str) -> dict | None:
    """
    Full picture of a doctor's given day: whether they work that weekday,
    their working hours, and their existing active bookings on that date.
    Used to explain WHY a doctor isn't free (outside hours vs. already
    booked) instead of a bare "not available" — e.g. "she works
    10:00-18:00, and 09:00 is before she opens" or "she's already booked
    at 14:00-14:30 that day".
    """
    async with aiosqlite.connect(DB_PATH) as db:
        resolved_name = await _resolve_doctor_name(db, doctor_name)
        if not resolved_name:
            return None
        async with db.execute(
            "SELECT name, available_days, start_time, end_time FROM doctors WHERE name = ?",
            (resolved_name,)
        ) as cursor:
            row = await cursor.fetchone()
        if not row:
            return None
        name, available_days, start_time, end_time = row
        weekday = datetime.strptime(appointment_date, "%Y-%m-%d").strftime("%A")
        works_that_day = weekday in [d.strip() for d in available_days.split(",")]

        async with db.execute("""
            SELECT a.appointment_time, a.duration_minutes
            FROM appointments a
            JOIN doctors d ON a.doctor_id = d.id
            WHERE d.name = ? AND a.appointment_date = ? AND a.status = 'active'
            ORDER BY a.appointment_time
        """, (resolved_name, appointment_date)) as cursor:
            bookings = [{"time": r[0], "duration": r[1]} for r in await cursor.fetchall()]

        return {
            "name": name,
            "available_days": available_days,
            "start_time": start_time,
            "end_time": end_time,
            "weekday": weekday,
            "works_that_day": works_that_day,
            "bookings": bookings,
        }


async def get_appointment_by_id(appointment_id: int) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("""
            SELECT a.id, a.appointment_date, a.appointment_time, a.duration_minutes,
                   a.status, a.patient_name, a.patient_email, a.phone_number,
                   a.reason_for_visit, a.sender_id, d.name AS doctor_name
            FROM appointments a
            JOIN doctors d ON a.doctor_id = d.id
            WHERE a.id = ?
        """, (appointment_id,)) as cursor:
            row = await cursor.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "appointment_date": row[1],
            "appointment_time": row[2],
            "duration_minutes": row[3],
            "status": row[4],
            "patient_name": row[5],
            "patient_email": row[6],
            "phone_number": row[7],
            "reason_for_visit": row[8],
            "sender_id": row[9],
            "doctor_name": row[10],
        }


async def get_appointments_by_sender(sender_id: str) -> list[dict]:
    """Return all active appointments for a given sender_id."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("""
            SELECT a.id, a.appointment_date, a.appointment_time, a.duration_minutes,
                   a.status, a.patient_name, a.patient_email, d.name AS doctor_name
            FROM appointments a
            JOIN doctors d ON a.doctor_id = d.id
            WHERE a.sender_id = ? AND a.status = 'active'
            ORDER BY a.appointment_date, a.appointment_time
        """, (sender_id,)) as cursor:
            rows = await cursor.fetchall()
        return [
            {
                "id": row[0],
                "appointment_date": row[1],
                "appointment_time": row[2],
                "duration_minutes": row[3],
                "status": row[4],
                "patient_name": row[5],
                "patient_email": row[6],
                "doctor_name": row[7],
            }
            for row in rows
        ]