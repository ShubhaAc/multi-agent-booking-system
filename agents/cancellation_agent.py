import logging
from langchain_openai import ChatOpenAI
from state import GraphState
from db import (
    cancel_appointment_with_ownership as db_cancel_with_ownership,
    get_appointment_by_id as db_get_appointment_by_id,
)
from tools.email_sender import send_cancellation_email
from config import MODEL_NAME
from agents.booking_agent import booking_node

logger = logging.getLogger(__name__)
llm = ChatOpenAI(model=MODEL_NAME, temperature=0)


async def cancellation_node(state: GraphState) -> dict:
    logger.info("Cancellation agent started.")

    if not state.appointment_id:
        return {"response_message": "Sure — could you give me the appointment ID you'd like to cancel? You can find it in your confirmation email."}

    if not state.sender_id:
        return {"response_message": "I need to verify your identity before cancelling. Could you share the email you used when booking?"}

    # Look up appointment
    appt = await db_get_appointment_by_id(state.appointment_id)
    if not appt:
        return {"response_message": f"I couldn't find an appointment with ID {state.appointment_id}. Please double-check the ID."}

    # Cancel with sender_id ownership check
    result = await db_cancel_with_ownership(state.appointment_id, state.sender_id)

    if not result["success"]:
        if result["reason"] == "not_owner":
            return {"response_message": f"Appointment {state.appointment_id} doesn't appear to belong to your account."}
        if result["reason"] == "already_cancelled":
            return {"response_message": f"Appointment {state.appointment_id} has already been cancelled."}
        return {"response_message": f"Could not cancel appointment {state.appointment_id}. Please try again or contact the clinic."}

    logger.info("Appointment %s cancelled by sender %s.", state.appointment_id, state.sender_id)

    # Send cancellation email
    try:
        await send_cancellation_email.ainvoke({
            "to_email": appt["patient_email"],
            "doctor_name": appt["doctor_name"],
            "booking_date": appt["appointment_date"],
            "booking_time": appt["appointment_time"],
            "appointment_id": state.appointment_id,
        })
        logger.info("Cancellation email sent to %s", appt["patient_email"])
        email_note = f" A cancellation confirmation has been sent to {appt['patient_email']}."
    except Exception as e:
        logger.error("Failed to send cancellation email: %s", e)
        email_note = ""

    cancel_response = (
        f"Your appointment with {appt['doctor_name']} on {appt['appointment_date']} "
        f"at {appt['appointment_time']} (ID: {state.appointment_id}) has been successfully cancelled."
        f"{email_note}"
    )

    # Chain into booking if rebook requested
    if state.rebook_requested:
        logger.info("Compound cancel-and-rebook — chaining into booking_node. doctor_name=%s", state.doctor_name)
        rebook_state = state.model_copy(update={
            "appointment_id": None,
            "cancellation_confirmed": True,
            "rebook_requested": False,
        })
        booking_update = await booking_node(rebook_state)
        booking_update["response_message"] = f"{cancel_response}\n\n{booking_update.get('response_message', '')}"
        booking_update["cancellation_confirmed"] = True
        booking_update["rebook_requested"] = False
        return booking_update

    return {
        "cancellation_confirmed": True,
        "appointment_id": None,
        "intent": None,
        "rebook_requested": False,
        "response_message": cancel_response,
    }