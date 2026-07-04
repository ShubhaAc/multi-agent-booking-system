import logging
import re
from difflib import SequenceMatcher
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from db.schema import init_db
from graph import build_graph
from state import GraphState
from store import get_conversation, save_conversation
from main import opik_tracer
from fastapi.middleware.cors import CORSMiddleware
from rag import upsert_chroma, upsert_pinecone, upsert_milvus


logger = logging.getLogger(__name__)
graph = None

SMALLTALK = {
    "hello", "hi", "hey", "hii", "hiii", "hello!", "hi!", "hey!",
    "good morning", "good afternoon", "good evening",
    "thanks", "thank you", "ok", "okay", "bye", "goodbye", "ciao"
}

SMALLTALK_REPLIES = {
    "hello": "Hello! Welcome to BrightSmile Dental Clinic. I can help you book, cancel, or reschedule appointments, check doctor availability, or answer questions about our services. How can I help?",
    "hi": "Hi there! How can I help you today?",
    "hey": "Hey! What can I do for you?",
    "hii": "Hi! How can I help?",
    "hiii": "Hi! How can I help?",
    "good morning": "Good morning! How can I help you today?",
    "good afternoon": "Good afternoon! How can I help you today?",
    "good evening": "Good evening! How can I help you today?",
    "thanks": "You're welcome! Let me know if there's anything else I can help with.",
    "thank you": "You're welcome! Is there anything else you need?",
    "ok": "Got it! Let me know if there's anything else.",
    "okay": "Got it! Let me know if there's anything else.",
    "bye": "Goodbye! Take care of your smile!",
    "goodbye": "Goodbye! Feel free to reach out anytime.",
    "ciao": "Ciao! Feel free to reach out anytime.",
}

_SMALLTALK_KEYS = list(SMALLTALK_REPLIES.keys())


def _closest_smalltalk(msg_lower: str) -> str | None:
    if not msg_lower or len(msg_lower.split()) > 3:
        return None
    cleaned = re.sub(r"[^a-z\s]", "", msg_lower)
    collapsed = re.sub(r"(.)\1+", r"\1", cleaned)
    best_key, best_ratio = None, 0.0
    for key in _SMALLTALK_KEYS:
        ratio = max(
            SequenceMatcher(None, cleaned, key).ratio(),
            SequenceMatcher(None, collapsed, key).ratio()
        )
        if ratio > best_ratio:
            best_ratio, best_key = ratio, key
    return best_key if best_ratio >= 0.72 else None


_HISTORY_ASSISTANT_MAX_CHARS = 220


def _compact_for_history(text: str) -> str:
    text = text.strip()
    if len(text) <= _HISTORY_ASSISTANT_MAX_CHARS:
        return text
    sentences = re.split(r"(?<=[.?!])\s+", text)
    last_sentence = sentences[-1].strip() if sentences else ""
    if last_sentence and len(last_sentence) < _HISTORY_ASSISTANT_MAX_CHARS - 40:
        head_budget = _HISTORY_ASSISTANT_MAX_CHARS - len(last_sentence) - 5
        head = text[:head_budget].rsplit(" ", 1)[0]
        return f"{head} […] {last_sentence}"
    truncated = text[:_HISTORY_ASSISTANT_MAX_CHARS].rsplit(" ", 1)[0]
    return f"{truncated} […]"


# Fields that should never be overwritten with None when merging result into prior.
# These are set once and must persist until explicitly cleared by agent logic.
_STICKY_FIELDS = {
    "sender_id", "patient_name", "phone_number", "invitee_email",
    "reason_for_visit", "specialization_needed", "doctor_name",
    "appointment_date", "appointment_time", "duration_minutes",
    "appointment_id", "suggested_alternative", "rebook_requested",
    "intent", "vector_store",
}


def _merge_state(prior: dict, result: dict) -> dict:
    """
    Merge graph result over prior Redis state.
    For sticky fields: only overwrite if the new value is not None.
    For non-sticky fields: always take the new value.
    This ensures fields not returned by a node (e.g. appointment_id
    when scheduling_node returns an early 'please provide X' message)
    are preserved in Redis for the next turn.
    """
    merged = dict(prior)
    for key, value in result.items():
        if key in _STICKY_FIELDS:
            if value is not None:
                merged[key] = value
        else:
            merged[key] = value
    return merged


@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph
    await init_db()
    graph = build_graph()
    logger.info("Dental clinic booking system ready.")
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    sender_id: str
    vector_store: str = "chroma"

    class Config:
        extra = "forbid"


class ChatResponse(BaseModel):
    response: str


@app.get("/")
async def root():
    return {"status": "ok", "message": "Dental clinic booking system API is running."}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    convo = await get_conversation(request.sender_id)
    history = convo["history"]
    prior = convo["state"]

    # Fast path — zero LLM calls for smalltalk with no active flow
    msg_lower = request.message.strip().lower().rstrip("!.,")
    matched_key = msg_lower if msg_lower in SMALLTALK else _closest_smalltalk(msg_lower)
    if matched_key and not prior.get("intent"):
        reply = SMALLTALK_REPLIES.get(matched_key, "Hello! How can I help you today?")
        history.append({"role": "user", "content": request.message})
        history.append({"role": "assistant", "content": _compact_for_history(reply)})
        await save_conversation(request.sender_id, history, prior)
        return ChatResponse(response=reply)

    initial_state = GraphState(
        user_message=request.message,
        history=history,
        vector_store=request.vector_store,
        sender_id=request.sender_id,
        intent=prior.get("intent"),
        invitee_email=prior.get("invitee_email"),
        doctor_name=prior.get("doctor_name"),
        appointment_date=prior.get("appointment_date"),
        appointment_time=prior.get("appointment_time"),
        duration_minutes=prior.get("duration_minutes", 30),
        patient_name=prior.get("patient_name"),
        phone_number=prior.get("phone_number"),
        reason_for_visit=prior.get("reason_for_visit"),
        specialization_needed=prior.get("specialization_needed"),
        suggested_alternative=prior.get("suggested_alternative"),
        appointment_id=prior.get("appointment_id"),
        rebook_requested=prior.get("rebook_requested", False),
    )

    result = await graph.ainvoke(
        initial_state,
        config={"callbacks": [opik_tracer]},
    )

    response = (result.get("response_message") or "").strip()
    reply = response or "Sorry, I did not understand that."

    history.append({"role": "user", "content": request.message})
    history.append({"role": "assistant", "content": _compact_for_history(reply)})

    # Merge result over prior — sticky fields only overwritten when non-None
    saved_state = _merge_state(prior, result)
    # Always stamp the sender_id so it's always in Redis
    saved_state["sender_id"] = request.sender_id

    await save_conversation(request.sender_id, history, saved_state)

    return ChatResponse(response=reply)


@app.delete("/reset/{sender_id}")
async def reset_conversation(sender_id: str):
    await save_conversation(sender_id, [], {})
    return {"status": "ok", "message": f"Conversation reset for {sender_id}"}


# Backend name -> the module-level function that clears that backend's
# existing index and re-ingests data/*.md fresh (see rag/*_store.py).
_VECTOR_STORE_UPSERT = {
    "chroma": upsert_chroma,
    "pinecone": upsert_pinecone,
    "milvus": upsert_milvus,
}


class VectorStoreUpdateRequest(BaseModel):
    sender_id: str
    vector_store: str  # "chroma" | "pinecone" | "milvus"

    class Config:
        extra = "forbid"


class VectorStoreUpdateResponse(BaseModel):
    status: str
    vector_store: str
    message: str


@app.post("/vectorstore/update", response_model=VectorStoreUpdateResponse)
async def update_vector_store(request: VectorStoreUpdateRequest):
    """
    Clear all existing chunks currently in the given backend's index and
    re-ingest fresh from data/*.md (clear-then-rebuild, not an incremental
    upsert), then remember this sender's backend choice so their next
    /chat call uses it. Does not touch conversation history — history
    lives in Redis, entirely independent of the vector store backend.
    """
    upsert_fn = _VECTOR_STORE_UPSERT.get(request.vector_store)
    if upsert_fn is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown vector_store '{request.vector_store}'. Choose one of: {', '.join(_VECTOR_STORE_UPSERT)}.",
        )

    # Embedding + index calls are blocking network/CPU work — run off the
    # event loop so this doesn't stall other in-flight /chat requests.
    logger.info("Clearing and rebuilding %s index for sender %s...", request.vector_store, request.sender_id)
    await run_in_threadpool(upsert_fn)

    convo = await get_conversation(request.sender_id)
    new_state = dict(convo["state"])
    new_state["vector_store"] = request.vector_store
    new_state["sender_id"] = request.sender_id
    await save_conversation(request.sender_id, convo["history"], new_state)

    logger.info("Vector store for %s set to %s.", request.sender_id, request.vector_store)
    return VectorStoreUpdateResponse(
        status="ok",
        vector_store=request.vector_store,
        message=f"Cleared and rebuilt the {request.vector_store} index with the latest documents.",
    )