import logging
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from state import GraphState
from tools import check_room_availability, book_room, send_invite_email, find_alternative
from config import MODEL_NAME

logger = logging.getLogger(__name__)

llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
booking_agent = create_react_agent(llm, tools=[check_room_availability, book_room, send_invite_email, find_alternative])

async def booking_node(state: GraphState) -> dict:
    logger.info("Booking agent started.")

    user_message = (
        f"You must follow these steps in order using only the tools provided:\n"
        f"Step 1: Call check_room_availability for {state.room_name} on {state.booking_date} "
        f"at {state.booking_time} for {state.duration_minutes} minutes.\n"       
        f"Step 2: If available, call book_room for {state.booked_by} with duration {state.duration_minutes} minutes. "
        f"Then call send_invite_email ONLY if invitee email is not None. Current invitee email: {state.invitee_email}.\n"
        f"Step 3: If NOT available, you MUST call find_alternative tool with room_name={state.room_name}, "
        f"booking_date={state.booking_date}, booking_time={state.booking_time}, duration_minutes={state.duration_minutes}. "
        f"Report exactly what the tool returns. Do not guess or make up room names."
    )

    result = await booking_agent.ainvoke({"messages": [{"role": "user", "content": user_message}]})
    response = result["messages"][-1].content
    logger.info("Booking agent response: %s", response)
    return {"response_message": response}