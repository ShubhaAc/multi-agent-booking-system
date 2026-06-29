import logging
import json
from datetime import date
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from state import GraphState
from config import MODEL_NAME

logger = logging.getLogger(__name__)
llm = ChatOpenAI(model=MODEL_NAME, temperature=0)

SYSTEM_PROMPT = """
You are a booking system supervisor. Extract intent and details from the user message.
Only assign an intent if the message is clearly about rooms or bookings.
Respond ONLY with a JSON object in this exact shape:
{{
    "intent": "book" | "cancel" | "reschedule" | "check_availability" | null,
    "room_name": string | null,
    "booking_date": string | null,
    "booking_time": string | null,
    "duration_minutes": integer | null,
    "invitee_email": string | null,
    "booking_id": integer | null
}}
Rules:
- Only use "check_availability" if the user is asking about a specific room or all rooms.
- If the message is unrelated to room booking, set intent to null.
- Always use the full room name as it appears in the message (e.g. "Nebula Room" not "Nebula").
- If no duration is mentioned, set duration_minutes to null.
- All dates must be in YYYY-MM-DD format. All times must be in HH:MM 24-hour format.
- Use the conversation history to resolve follow-up messages. For example if the user previously asked about a room and now says "book it", extract the room name and details from the history.
Today's date is {today}.

Conversation history:
{history}
"""

async def supervisor_node(state: GraphState) -> dict:
    logger.info("Supervisor processing message: %s", state.user_message)
    today = date.today().isoformat()

    history_text = ""
    for msg in state.history:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT.format(today=today, history=history_text or "No history yet.")),
        HumanMessage(content=state.user_message)
    ])
    parsed = json.loads(response.content)
    logger.info("Supervisor extracted: %s", parsed)
    return {
        "intent": parsed.get("intent"),
        "room_name": parsed.get("room_name"),
        "booking_date": parsed.get("booking_date"),
        "booking_time": parsed.get("booking_time"),
        "duration_minutes": parsed.get("duration_minutes") or 60,
        "invitee_email": parsed.get("invitee_email"),
        "booking_id": parsed.get("booking_id"),
    }