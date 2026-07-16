import logging
import math
import re
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from state import GraphState
from rag import get_chroma_retriever, get_pinecone_retriever
from store import get_faq_index, add_faq_entry
from config import MODEL_NAME, FAQ_EMBEDDING_MODEL, FAQ_SIMILARITY_THRESHOLD, FAQ_INDEX_MAX_ENTRIES

logger = logging.getLogger(__name__)

# Reduced to 256 dims (OpenAI v3 embedding model)
_embeddings = OpenAIEmbeddings(model=FAQ_EMBEDDING_MODEL, dimensions=256)


_PERSONAL_PRONOUN_RE = re.compile(r"\b(my|i'?m|i|mine|our|we|me|myself)\b", re.IGNORECASE)


def _is_cacheable(question: str) -> bool:
    return not _PERSONAL_PRONOUN_RE.search(question)


_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load_full_document(filename: str) -> str:
    path = _DATA_DIR / filename
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        logger.warning("Could not read %s for full-context fallback.", path)
        return ""


# ---------------------------------------------------------------------------
# Deterministic routing for service/treatment questions.
#
# OVERVIEW  - "what services do you offer?", "what treatments do you have?"
#             -> hard-coded category summary, parsed from services.md's
#                headings. No LLM call, no retrieval — so there's no
#                context in play for the model to blend across files
#                (this is what let doctor names from doctors.md bleed in).
#
# DETAIL    - "show all services", "give me the complete list", "full list"
#             -> raw services.md content, returned as-is. Still no LLM call:
#                the model was previously being told not to summarize/omit
#                anything, which is exactly what "return the file" does
#                deterministically without the hallucination risk.
#
# Both bypass the semantic FAQ cache too — they're already O(1) and
# deterministic, so caching adds risk (a differently-phrased, personalized
# question matching by embedding similarity later) with no upside.
#
# DETAIL is checked before OVERVIEW: "what are all the services you offer"
# matches both patterns, and "all" is the more specific signal.
# ---------------------------------------------------------------------------

_SERVICES_DETAIL_PATTERN = re.compile(
    r"\ball\s+(the\s+|your\s+)?(services|treatments)\b"
    r"|\bcomplete\s+list\b"
    r"|\bfull\s+list\b"
    r"|\bshow\s+(me\s+)?all\b"
    r"|\bevery\s+(service|treatment)\b"
    r"|\bfull\s+(range|details?)\b",
    re.IGNORECASE,
)

_SERVICES_OVERVIEW_PATTERN = re.compile(
    r"\b(what|which)\s+(services|treatments)\b.{0,20}\b(offer|provide|have|available)\b"
    r"|\bwhat\s+do\s+you\s+offer\b"
    r"|\brange\s+of\s+(services|treatments)\b"
    r"|^\s*services\s*\??\s*$"
    r"|^\s*treatments\s*\??\s*$",
    re.IGNORECASE,
)

_SERVICES_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def _build_services_overview(markdown: str) -> str:
    """Category-only summary, parsed from services.md's ## headings so this
    can never drift out of sync with the actual service list."""
    categories = _SERVICES_HEADING_RE.findall(markdown)
    if not categories:
        return (
            "We offer a full range of dental services. Ask me about a "
            "specific treatment and I can tell you more!"
        )
    bullets = "\n".join(f"- {c}" for c in categories)
    return (
        "Here's an overview of what we offer:\n\n"
        f"{bullets}\n\n"
        "Want the full list of treatments under any of these, or details "
        "on something specific? Just ask!"
    )


async def _semantic_cache_lookup(question: str) -> str | None:
    """Embed the question and compare it against every cached FAQ entry by
    cosine similarity. This is what actually handles "what services do you
    provide" vs "what are the services" vs "what do you offer" — an exact or
    normalized string match would treat those as three different questions
    and miss the cache every time; embeddings put them close together in
    vector space regardless of exact wording."""
    index = await get_faq_index()
    if not index:
        return None

    query_embedding = await _embeddings.aembed_query(question)

    best_score, best_answer, best_question = 0.0, None, None
    for entry in index:
        score = _cosine_similarity(query_embedding, entry["embedding"])
        if score > best_score:
            best_score, best_answer, best_question = score, entry["answer"], entry["question"]

    if best_score >= FAQ_SIMILARITY_THRESHOLD:
        logger.info(
            "FAQ semantic cache HIT (score=%.3f >= %.2f): %r matched cached %r",
            best_score, FAQ_SIMILARITY_THRESHOLD, question, best_question,
        )
        return best_answer

    logger.info(
        "FAQ semantic cache MISS (best score=%.3f < %.2f, closest was %r)",
        best_score, FAQ_SIMILARITY_THRESHOLD, best_question,
    )
    return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def _known_context(state: GraphState) -> str:
    """Known booking fields, compact form, so follow-ups like "price for this",
    "my doctor", or "what's my name" resolve without the user repeating
    themselves. patient_name/phone/email were missing here before — the
    knowledge agent had no way to answer identity questions even though
    that data was already sitting in state."""
    fields = [
        state.patient_name and f"patient_name={state.patient_name}",
        state.phone_number and f"phone_number={state.phone_number}",
        state.invitee_email and f"email={state.invitee_email}",
        state.doctor_name and f"doctor={state.doctor_name}",
        state.reason_for_visit and f"reason={state.reason_for_visit}",
        state.specialization_needed and f"specialization={state.specialization_needed}",
        state.appointment_date and f"date={state.appointment_date}",
        state.appointment_time and f"time={state.appointment_time}",
        state.appointment_id and f"appointment_id={state.appointment_id}",
    ]
    known = [f for f in fields if f]
    return ", ".join(known) if known else "none yet"


def _recent_history(state: GraphState, limit: int = 4) -> str:
    recent = state.history[-limit:] if len(state.history) > limit else state.history
    if not recent:
        return "none"
    return "\n".join(f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in recent)


async def knowledge_node(state: GraphState) -> dict:
    logger.info("Knowledge agent started.")

    # Deterministic service-query routing runs first, before caching, RAG,
    # or any LLM call — so broad "what do you offer" questions can never
    # pick up unrelated context (e.g. a doctor's name) from other documents.
    if _SERVICES_DETAIL_PATTERN.search(state.user_message):
        logger.info("Services-detail intent detected — returning raw services.md, no LLM.")
        full_doc = _load_full_document("services.md")
        if full_doc:
            return {"response_message": full_doc.strip()}
        # fall through to normal handling below if the file couldn't be read
    elif _SERVICES_OVERVIEW_PATTERN.search(state.user_message):
        logger.info("Services-overview intent detected — returning category summary, no LLM.")
        full_doc = _load_full_document("services.md")
        return {"response_message": _build_services_overview(full_doc)}

    cacheable = _is_cacheable(state.user_message)
    if cacheable:
        cached = await _semantic_cache_lookup(state.user_message)
        if cached:
            return {"response_message": cached}

    known_context = _known_context(state)
    recent_history = _recent_history(state)

    if state.vector_store == "pinecone":
        logger.info("Using Pinecone retriever.")
        retriever = get_pinecone_retriever()
    else:
        logger.info("Using Chroma retriever.")
        retriever = get_chroma_retriever()

    retrieval_query = " ".join(
        p for p in (state.user_message, state.doctor_name, state.reason_for_visit) if p
    )
    docs = await retriever.ainvoke(retrieval_query)
    context = format_docs(docs)

    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    prompt = ChatPromptTemplate.from_template("""
You are a helpful dental clinic assistant. Answer using the context and known details below.
Resolve "this/it/my doctor/my name" using Known Details or Recent Conversation before saying you don't know.

Context:
{context}

Known Details: {known_context}
Recent Conversation:
{recent_history}

Question: {question}
""")
    chain = prompt | llm | StrOutputParser()
    response = await chain.ainvoke({
        "context": context,
        "known_context": known_context,
        "recent_history": recent_history,
        "question": state.user_message,
    })

    logger.info("Knowledge agent response: %s", response)

    if cacheable:
        answer_embedding = await _embeddings.aembed_query(state.user_message)
        await add_faq_entry(state.user_message, response, answer_embedding, FAQ_INDEX_MAX_ENTRIES)

    return {"response_message": response}