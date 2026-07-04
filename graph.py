import logging
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from state import GraphState
from agents import supervisor_node, booking_node, cancellation_node, scheduling_node, knowledge_node
from config import MODEL_NAME

logger = logging.getLogger(__name__)


async def fallback_node(state: GraphState) -> dict:
    """
    Safety net only. Normally the supervisor already fills response_message
    directly (via fallback_response) when intent is null, so this node is
    skipped entirely — see route_intent. This only runs if the supervisor
    failed to produce a fallback_response (e.g. malformed JSON), so it's rare.
    """
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0.4)
    response = await llm.ainvoke([
        SystemMessage(content=(
            "You are a friendly front-desk assistant at BrightSmile Dental Clinic. "
            "Respond naturally and warmly to the user's message. "
            "If they're greeting you, greet them back and briefly mention you can help with "
            "booking, cancelling, or rescheduling appointments, checking doctor availability, "
            "or answering questions about the clinic. "
            "Keep it short — 1 to 3 sentences max. No bullet points."
        )),
        HumanMessage(content=state.user_message)
    ])
    return {"response_message": response.content.strip()}


def route_intent(state: GraphState) -> str:
    logger.info("Routing intent: %s", state.intent)
    if state.intent == "book":
        return "booking"
    elif state.intent == "cancel":
        return "cancellation"
    elif state.intent in ("reschedule", "check_availability"):
        return "scheduling"
    elif state.intent == "knowledge":
        return "knowledge"
    elif state.intent is None:
        # Supervisor already answered directly (fallback_response) — skip the
        # extra LLM call. Only fall through to fallback_node if it didn't.
        if state.response_message:
            return "done"
        return "fallback"
    else:
        logger.warning("Unknown intent: %s — routing to fallback", state.intent)
        return "fallback"


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("booking", booking_node)
    graph.add_node("cancellation", cancellation_node)
    graph.add_node("scheduling", scheduling_node)
    graph.add_node("knowledge", knowledge_node)
    graph.add_node("fallback", fallback_node)
    graph.set_entry_point("supervisor")
    graph.add_conditional_edges("supervisor", route_intent, {
        "booking": "booking",
        "cancellation": "cancellation",
        "scheduling": "scheduling",
        "knowledge": "knowledge",
        "fallback": "fallback",
        "done": END,
    })
    graph.add_edge("booking", END)
    graph.add_edge("cancellation", END)
    graph.add_edge("scheduling", END)
    graph.add_edge("knowledge", END)
    graph.add_edge("fallback", END)
    return graph.compile()