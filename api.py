import logging
from contextlib import asynccontextmanager
from collections import defaultdict

from fastapi import FastAPI
from pydantic import BaseModel

from db.schema import init_db
from graph import build_graph
from state import GraphState

logger = logging.getLogger(__name__)

graph = None
conversation_history: dict[str, list[dict]] = defaultdict(list)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph
    await init_db()
    graph = build_graph()
    logger.info("Booking system ready.")
    yield

app = FastAPI(lifespan=lifespan)

class ChatRequest(BaseModel):
    message: str
    user_email: str = "guest@booking.com"

class ChatResponse(BaseModel):
    response: str

@app.get("/")
async def root():
    return {"status": "ok", "message": "Booking system API is running."}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    history = conversation_history[request.user_email]

    initial_state = GraphState(
        user_message=request.message,
        booked_by=request.user_email,
        history=history,
    )

    result = await graph.ainvoke(initial_state)
    response = result.get("response_message", "").strip()
    reply = response or "Sorry, I did not understand that."

    # Update history
    conversation_history[request.user_email].append({"role": "user", "content": request.message})
    conversation_history[request.user_email].append({"role": "assistant", "content": reply})

    return ChatResponse(response=reply)