import logging
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from state import GraphState
from tools import cancel_room_booking
from config import MODEL_NAME

logger = logging.getLogger(__name__)

llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
cancellation_agent = create_react_agent(llm, tools=[cancel_room_booking])

async def cancellation_node(state: GraphState) -> dict:
    logger.info("Cancellation agent started.")
    user_message = f"Cancel booking with ID {state.booking_id}."
    result = await cancellation_agent.ainvoke({"messages": [{"role": "user", "content": user_message}]})
    response = result["messages"][-1].content
    logger.info("Cancellation agent response: %s", response)
    return {"cancellation_confirmed": True, "response_message": response}