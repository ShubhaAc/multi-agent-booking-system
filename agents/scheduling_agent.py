import logging
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from state import GraphState
from tools import find_alternative, check_room_availability, list_available_rooms
from config import MODEL_NAME

logger = logging.getLogger(__name__)

llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
scheduling_agent = create_react_agent(llm, tools=[find_alternative, check_room_availability, list_available_rooms])

async def scheduling_node(state: GraphState) -> dict:
    logger.info("Scheduling agent started.")

    if state.intent == "check_availability":
        if not state.room_name:
            user_message = (
                f"List all available rooms on {state.booking_date} at {state.booking_time} "
                f"for {state.duration_minutes} minutes. Use the list_available_rooms tool."
            )
        elif not state.booking_time:
            user_message = (
                f"The user asked if {state.room_name} is available but did not specify a time. "
                f"Do not call any tools. Just reply: "
                f"'Please specify a date and time to check availability for {state.room_name}.'"
            )
        else:
            user_message = (
                f"Check if {state.room_name} is available on {state.booking_date} "
                f"at {state.booking_time} for {state.duration_minutes} minutes. "
                f"Use check_room_availability tool only. Report exactly what the tool returns."
            )
    else:
        user_message = (
            f"Find an alternative room since {state.room_name} is unavailable "
            f"on {state.booking_date} at {state.booking_time} for {state.duration_minutes} minutes. "
            f"Use find_alternative tool."
        )

    result = await scheduling_agent.ainvoke({"messages": [{"role": "user", "content": user_message}]})
    response = result["messages"][-1].content
    logger.info("Scheduling agent response: %s", response)
    return {"response_message": response}