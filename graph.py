import logging
from langgraph.graph import StateGraph, END
from state import GraphState
from agents import supervisor_node, booking_node, cancellation_node, scheduling_node

logger = logging.getLogger(__name__)

async def fallback_node(state: GraphState) -> dict:
    return {"response_message": "I'm not sure what you want to do. I can help you book, cancel, reschedule, or check room availability."}

def route_intent(state: GraphState) -> str:
    logger.info("Routing intent: %s", state.intent)
    if state.intent == "book":
        return "booking"
    elif state.intent == "cancel":
        return "cancellation"
    elif state.intent in ("reschedule", "check_availability"):
        return "scheduling"
    else:
        logger.warning("Unknown intent: %s — routing to fallback", state.intent)
        return "fallback"

def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("booking", booking_node)
    graph.add_node("cancellation", cancellation_node)
    graph.add_node("scheduling", scheduling_node)
    graph.add_node("fallback", fallback_node)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges("supervisor", route_intent, {
        "booking": "booking",
        "cancellation": "cancellation",
        "scheduling": "scheduling",
        "fallback": "fallback"
    })

    graph.add_edge("booking", END)
    graph.add_edge("cancellation", END)
    graph.add_edge("scheduling", END)
    graph.add_edge("fallback", END)

    return graph.compile()