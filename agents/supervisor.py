import logging
import re
from datetime import date, timedelta
from typing import Literal, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from state import GraphState
from config import MODEL_NAME
from skillopt.skill_store import load_active_skill


logger = logging.getLogger(__name__)
llm = ChatOpenAI(model=MODEL_NAME, temperature=0, seed=42)

class SupervisorOutput(BaseModel):
    intent: Literal["book", "cancel", "reschedule", "check_availability", "knowledge", None] = Field(
        default=None,
        description="User intent. Null only if unrelated to clinic AND nothing in progress.",
    )
    doctor_name: Optional[str] = Field(
        default=None,
        description="Doctor named in THIS message only, never from history. Null = carry forward. Named doctor beats clear_doctor.",
    )
    clear_doctor: bool = Field(
        default=False,
        description="True = wipe pinned doctor, no replacement named (e.g. 'whoever's free'). Only for book/reschedule/cancel; never with doctor_name set.",
    )
    appointment_date: Optional[str] = Field(
        default=None,
        description="Explicit date as YYYY-MM-DD. Null if relative/weekday phrase (use relative_date_phrase instead).",
    )
    relative_date_phrase: Optional[str] = Field(
        default=None,
        description="Relative date phrase verbatim (e.g. 'thursday', 'tomorrow'). Do not resolve to a date yourself.",
    )
    appointment_time: Optional[str] = Field(default=None, description="Time, 24h HH:MM.")
    duration_minutes: Optional[int] = Field(default=None, description="Duration in minutes, if stated this turn.")
    patient_name: Optional[str] = Field(default=None, description="Patient name from this message only. Never an email.")
    phone_number: Optional[str] = Field(default=None, description="Phone number from this message.")
    reason_for_visit: Optional[str] = Field(default=None, description="Visit reason from this message.")
    specialization_needed: Optional[str] = Field(
        default=None,
        description="Short specialization label only (e.g. 'General', 'Cosmetic'), matched against DB — not the full clinical name.",
    )
    invitee_email: Optional[str] = Field(default=None, description="Email from this message.")
    appointment_id: Optional[int] = Field(
        default=None,
        description="Integer ID ONLY if user explicitly states it this turn (e.g. '#7'). Never inferred/remembered; code handles carry-forward.",
    )
    rebook_requested: bool = Field(
        default=False,
        description="True only if user wants to cancel AND rebook in the same message.",
    )
    fallback_response: Optional[str] = Field(
        default=None,
        description="Only when intent is null: 1-2 sentence front-desk reply mentioning booking/cancel/reschedule/availability/clinic Q&A help. Else null.",
    )


structured_llm = llm.with_structured_output(SupervisorOutput, include_raw=True)


_call_stats = {"fast_path": 0, "llm_call": 0, "cached_tokens_total": 0, "prompt_tokens_total": 0}


def _log_llm_usage(response) -> None:
    """Pull token/cache usage off the LLM response and log it.

    OpenAI returns prompt_tokens_details.cached_tokens on cache hits, surfaced
    by langchain-openai in response.usage_metadata (and mirrored in
    response.response_metadata['token_usage']). Anthropic models expose the
    equivalent via cache_read_input_tokens in usage_metadata — this reads
    whichever fields are present so it works either way.
    """
    usage = getattr(response, "usage_metadata", None) or {}
    prompt_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or 0

    cached_tokens = 0
    details = usage.get("input_token_details") or {}
    if details:
        cached_tokens = details.get("cache_read") or details.get("cached_tokens") or 0
    if not cached_tokens:
        raw_details = (
            getattr(response, "response_metadata", {})
            .get("token_usage", {})
            .get("prompt_tokens_details", {})
        )
        cached_tokens = raw_details.get("cached_tokens", 0)

    _call_stats["llm_call"] += 1
    _call_stats["prompt_tokens_total"] += prompt_tokens
    _call_stats["cached_tokens_total"] += cached_tokens

    cache_pct = (cached_tokens / prompt_tokens * 100) if prompt_tokens else 0
    logger.info(
        "Supervisor LLM call #%d: prompt_tokens=%d cached_tokens=%d (%.0f%% cached) | "
        "running totals: fast_path_skips=%d llm_calls=%d avg_cache_rate=%.0f%%",
        _call_stats["llm_call"], prompt_tokens, cached_tokens, cache_pct,
        _call_stats["fast_path"], _call_stats["llm_call"],
        (_call_stats["cached_tokens_total"] / _call_stats["prompt_tokens_total"] * 100)
        if _call_stats["prompt_tokens_total"] else 0,
    )


''' PROMPT CACHING: this block is split in two on purpose.

- STATIC_INSTRUCTIONS never changes — no {placeholders}, byte-for-byte
- identical on every call, forever. OpenAI's prompt cache recognizes it after
- the first call and charges roughly half price for it on every call after

- DYNAMIC_CONTEXT_TEMPLATE holds everything that changes turn-to-turn (prev
  state, today's date, recent history). 
'''

STATIC_INSTRUCTIONS = load_active_skill("supervisor")


DYNAMIC_CONTEXT_TEMPLATE = """
# Previous State
intent={prev_intent} | doctor_name={prev_doctor_name} | suggested_alternative={prev_suggested_alternative}
appointment_id={prev_appointment_id}
date={prev_appointment_date} | time={prev_appointment_time} | duration={prev_duration_minutes}
patient={prev_patient_name} | phone={prev_phone_number} | reason={prev_reason_for_visit}
specialization={prev_specialization_needed} | email={prev_invitee_email}

Today: {today} ({today_weekday})

Recent conversation:
{history}
"""


_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

_WEEKDAY_WORD_PATTERN = re.compile(
    r'\b(next\s+|this\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
    re.IGNORECASE,
)
_EXPLICIT_DATE_PATTERN = re.compile(r'\d{1,4}[-/]\d{1,2}([-/]\d{1,4})?')
_MONTH_NAME_DATE_PATTERN = re.compile(
    r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}\b',
    re.IGNORECASE,
)

_AFFIRMATIVE_PATTERN = re.compile(
    r'^\s*(ok(ay)?|yes|yeah|yep|sure|go ahead|sounds good|confirm(ed)?)\s*[!.]*\s*$',
    re.IGNORECASE,
)

_REJECTION_PATTERN = re.compile(
    r"^\s*(no|nope|nah|cancel( that)?|stop|don'?t( do that)?)\s*[!.]*\s*$",
    re.IGNORECASE,
)

_ANY_DOCTOR_PATTERN = re.compile(
    r"^\s*(whoever'?s? (is )?free|any(one|body)?( doctor)?|no preference|"
    r"doesn'?t matter|don'?t (know|mind|care)|dont know|dunno|first available)\s*[!.]*\s*$",
    re.IGNORECASE,
)

_BOOKING_STATE_QUESTION_PATTERN = re.compile(
    r"\b(am i|was i|did i|is my|do i have).{0,25}\b(book|booked|appointment|schedule|scheduled)\b",
    re.IGNORECASE,
)
_DOCTOR_CHANGE_SIGNAL_PATTERN = re.compile(
    r"\b(another|different|someone else|switch|change (my |the )?doctor|new doctor)\b",
    re.IGNORECASE,
)

_DOCTOR_PREFERENCE_SIGNAL_PATTERN = re.compile(
    r"\b(experience[d]?|senior|junior|better|best|recommend(ed)?|expert|"
    r"first (one|option|doctor)|second (one|option|doctor)|the other (one|doctor)|"
    r"more (senior|junior|experienced)|less experienced|newer|older)\b",
    re.IGNORECASE,
)


def _extract_weekday_phrase(message: str) -> str | None:
    match = _WEEKDAY_WORD_PATTERN.search(message)
    if not match:
        return None
    prefix = (match.group(1) or "").strip().lower()
    day = match.group(2).lower()
    return f"{prefix} {day}".strip() if prefix else day


def _resolve_relative_date(phrase: str, today: date) -> str | None:
    if not phrase:
        return None
    p = phrase.strip().lower()
    if p == "today":
        return today.isoformat()
    if p == "tomorrow":
        return (today + timedelta(days=1)).isoformat()

    is_next = p.startswith("next ")
    day_word = p[5:].strip() if is_next else (p[5:].strip() if p.startswith("this ") else p)
    if day_word not in _WEEKDAYS:
        return None

    today_idx = today.weekday()
    target_idx = _WEEKDAYS[day_word]
    days_ahead = (target_idx - today_idx) % 7
    if days_ahead == 0:
        days_ahead = 7
    if is_next:
        days_ahead += 7
    return (today + timedelta(days=days_ahead)).isoformat()


_TITLE_WORDS = {"dr", "dr.", "doctor"}


def _doctor_mentioned_in_message(doctor_name: str, message: str) -> bool:
  
    if not doctor_name:
        return False
    msg_lower = message.lower()
    name_lower = doctor_name.lower()
    if name_lower in msg_lower:
        return True
    words = [w.strip(".,") for w in name_lower.split() if w.strip(".,") not in _TITLE_WORDS]
    return any(len(w) >= 3 and w in msg_lower for w in words)


_ECHO_CHECK_FIELDS = ("patient_name", "phone_number", "invitee_email")

def _looks_like_echoed_history(value: str, current_message: str, history: list[dict]) -> bool:
    if not value:
        return False
    normalized = value.strip().lower()
    if normalized in current_message.strip().lower():
        return False
    return any(
        msg.get("role") == "user" and (msg.get("content") or "").strip().lower() == normalized
        for msg in history
    )


def _carry_forward(parsed: dict, field: str, prev_value):
    value = parsed.get(field)
    return value if value is not None else prev_value


_CLEAR_ALLOWED_INTENTS = {"book", "reschedule", "cancel"}


async def supervisor_node(state: GraphState) -> dict:
    logger.info("Supervisor processing message: %s", state.user_message)
    today_date = date.today()

    
    if _AFFIRMATIVE_PATTERN.match(state.user_message) and state.suggested_alternative:
        logger.info(
            "Fast-path: affirmative confirmation of suggested_alternative=%r — skipping LLM call.",
            state.suggested_alternative,
        )
        _call_stats["fast_path"] += 1
        return {
            "intent": state.intent,
            "doctor_name": state.suggested_alternative,
            "appointment_date": state.appointment_date,
            "appointment_time": state.appointment_time,
            "duration_minutes": state.duration_minutes or 30,
            "patient_name": state.patient_name,
            "phone_number": state.phone_number,
            "reason_for_visit": state.reason_for_visit,
            "specialization_needed": state.specialization_needed,
            "invitee_email": state.invitee_email,
            "appointment_id": state.appointment_id,
            "suggested_alternative": None,
            "recommended_doctors": [],
            "rebook_requested": False,
        }

    if _REJECTION_PATTERN.match(state.user_message):
        logger.info("Fast-path: bare rejection (%r) — skipping LLM call.", state.user_message)
        _call_stats["fast_path"] += 1
        preserved_appointment_id = state.appointment_id if state.intent == "reschedule" else None
        return {
            "intent": None,
            "doctor_name": None,
            "appointment_date": None,
            "appointment_time": None,
            "suggested_alternative": None,
            "recommended_doctors": [],
            "appointment_id": preserved_appointment_id,
            "duration_minutes": state.duration_minutes or 30,
            "patient_name": state.patient_name,
            "phone_number": state.phone_number,
            "reason_for_visit": state.reason_for_visit,
            "specialization_needed": state.specialization_needed,
            "invitee_email": state.invitee_email,
            "rebook_requested": False,
            "response_message": "No problem — that's been cancelled. Let me know if there's anything else I can help with.",
        }

    # Normal path
    today = today_date.isoformat()
    today_weekday = today_date.strftime("%A")

    recent = state.history[-4:] if len(state.history) > 4 else state.history
    history_text = ""
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    dynamic_context = DYNAMIC_CONTEXT_TEMPLATE.format(
        today=today,
        today_weekday=today_weekday,
        history=history_text or "No history yet.",
        prev_intent=state.intent,
        prev_doctor_name=state.doctor_name,
        prev_suggested_alternative=state.suggested_alternative,
        prev_appointment_id=state.appointment_id,
        prev_appointment_date=state.appointment_date,
        prev_appointment_time=state.appointment_time,
        prev_duration_minutes=state.duration_minutes,
        prev_patient_name=state.patient_name,
        prev_phone_number=state.phone_number,
        prev_reason_for_visit=state.reason_for_visit,
        prev_specialization_needed=state.specialization_needed,
        prev_invitee_email=state.invitee_email,
    )

    response = await structured_llm.ainvoke([
        SystemMessage(content=STATIC_INSTRUCTIONS),   
        SystemMessage(content=dynamic_context),        
        HumanMessage(content=state.user_message),
    ])

   
    raw_message = response["raw"]
    parsed_output: SupervisorOutput = response["parsed"]

    _log_llm_usage(raw_message)

    parsed = parsed_output.model_dump()
    logger.info("Supervisor extracted: %s", parsed)

    for field in _ECHO_CHECK_FIELDS:
        value = parsed.get(field)
        if value and _looks_like_echoed_history(value, state.user_message, state.history):
            logger.warning(
                "Discarding %s=%r — matches a prior raw user message verbatim, likely echoed from history rather than extracted from the current message.",
                field, value,
            )
            parsed[field] = None

    for field, value in parsed.items():
        if isinstance(value, str) and value.strip().lower() == "null":
            logger.warning(
                "Discarding %s=%r — model output the literal string 'null' instead of an empty value.",
                field, value,
            )
            parsed[field] = None


    if parsed.get("intent") is None and _BOOKING_STATE_QUESTION_PATTERN.search(state.user_message):
            logger.warning(
                "Booking-status question classified as null intent — overriding to "
                "'knowledge' so it's answered from real state, not a generated fallback_response. "
                "message=%r", state.user_message,
            )
            parsed["intent"] = "knowledge"
            parsed["fallback_response"] = None

            

    if _REJECTION_PATTERN.match(state.user_message):
        logger.info("Detected explicit rejection (%r) — wiping proposal state.", state.user_message)
        preserved_appointment_id = state.appointment_id if state.intent == "reschedule" else None
        fallback_response = parsed.get("fallback_response")
        return {
            "intent": None,
            "doctor_name": None,
            "appointment_date": None,
            "appointment_time": None,
            "suggested_alternative": None,
            "appointment_id": preserved_appointment_id,
            "duration_minutes": state.duration_minutes or 30,
            "patient_name": state.patient_name,
            "phone_number": state.phone_number,
            "reason_for_visit": state.reason_for_visit,
            "specialization_needed": state.specialization_needed,
            "invitee_email": state.invitee_email,
            "rebook_requested": False,
            "response_message": fallback_response or "No problem — that's been cancelled. Let me know if there's anything else I can help with.",
        }

    resolved_intent = parsed.get("intent") if parsed.get("intent") is not None else state.intent

    resolved_appointment_id = parsed.get("appointment_id") if parsed.get("appointment_id") is not None else state.appointment_id

    if resolved_intent == "book" and state.intent != "book":
        logger.info("Fresh book intent detected — clearing stale appointment_id %s from prior flow.", resolved_appointment_id)
        resolved_appointment_id = None

    if resolved_intent == "reschedule" and not resolved_appointment_id:
        logger.info("Reschedule requested with no appointment_id yet — routing to scheduling to ask for it.")

    raw_doctor_name = parsed.get("doctor_name")
    if parsed.get("clear_doctor"):
        
        if raw_doctor_name and _doctor_mentioned_in_message(raw_doctor_name, state.user_message):
            logger.info(
                "clear_doctor=True but %r is genuinely named in message %r — treating as a replacement pick, not a clear.",
                raw_doctor_name, state.user_message,
            )
        else:
            if raw_doctor_name:
                logger.warning(
                    "clear_doctor=True and doctor_name=%r was set but not actually mentioned in "
                    "message %r — discarding the echoed name and honoring the clear.",
                    raw_doctor_name, state.user_message,
                )
            raw_doctor_name = "CLEAR"

    if (
        raw_doctor_name is None
        and resolved_intent in _CLEAR_ALLOWED_INTENTS
        and _ANY_DOCTOR_PATTERN.match(state.user_message)
    ):
        logger.info("Detected 'any doctor' phrasing — routing through CLEAR sentinel.")
        raw_doctor_name = "CLEAR"

 
    if (
        raw_doctor_name
        and raw_doctor_name != "CLEAR"
        and len(state.recommended_doctors) > 1
        and not _doctor_mentioned_in_message(raw_doctor_name, state.user_message)
        and not _DOCTOR_PREFERENCE_SIGNAL_PATTERN.search(state.user_message)
    ):
        logger.warning(
            "Discarding hallucinated doctor_name=%r — not mentioned in message %r, "
            "and %d candidates (%s) are still unresolved. Leaving doctor unpicked "
            "so the patient is asked to choose explicitly.",
            raw_doctor_name, state.user_message, len(state.recommended_doctors),
            state.recommended_doctors,
        )
        raw_doctor_name = None

    if raw_doctor_name == "CLEAR":
        if resolved_intent in _CLEAR_ALLOWED_INTENTS:
            spurious_clear = (
                resolved_intent == "reschedule"
                and state.doctor_name
                and not _DOCTOR_CHANGE_SIGNAL_PATTERN.search(state.user_message)
                and not _ANY_DOCTOR_PATTERN.match(state.user_message)
            )
            if spurious_clear:
                logger.warning(
                    "Ignoring spurious CLEAR sentinel — no doctor-change language in "
                    "%r during reschedule; keeping doctor_name=%r.",
                    state.user_message, state.doctor_name,
                )
                resolved_doctor_name = state.doctor_name
                resolved_suggested_alternative = state.suggested_alternative
            else:
                resolved_doctor_name = None
                resolved_suggested_alternative = None
                if resolved_intent != "reschedule":
                    resolved_appointment_id = None
        else:
            logger.warning(
                "Ignoring CLEAR doctor_name sentinel because intent=%s (not a booking context)",
                resolved_intent,
            )
            resolved_doctor_name = state.doctor_name
            resolved_suggested_alternative = state.suggested_alternative
    elif raw_doctor_name is not None:
        resolved_doctor_name = raw_doctor_name
        resolved_suggested_alternative = state.suggested_alternative
    else:
        resolved_doctor_name = state.doctor_name
        resolved_suggested_alternative = state.suggested_alternative

    if (
        _AFFIRMATIVE_PATTERN.match(state.user_message)
        and state.suggested_alternative
        and raw_doctor_name is None
    ):
        resolved_doctor_name = state.suggested_alternative
        resolved_suggested_alternative = None

    rebook_requested = bool(parsed.get("rebook_requested", False))

    relative_phrase = parsed.get("relative_date_phrase")

    weekday_phrase = _extract_weekday_phrase(state.user_message)
    has_explicit_date_signal = bool(
        _EXPLICIT_DATE_PATTERN.search(state.user_message)
    ) or bool(_MONTH_NAME_DATE_PATTERN.search(state.user_message))

    
    if (
        parsed.get("appointment_date")
        and not relative_phrase
        and not weekday_phrase
        and not has_explicit_date_signal
        and parsed.get("appointment_date") != state.appointment_date
    ):
        logger.warning(
            "Discarding unprompted appointment_date=%r — message %r has no date "
            "reference at all; keeping carried-forward date=%r.",
            parsed.get("appointment_date"), state.user_message, state.appointment_date,
        )
        parsed["appointment_date"] = None

    if relative_phrase and parsed.get("appointment_date"):
        logger.warning(
            "Model set BOTH appointment_date=%r and relative_date_phrase=%r for "
            "message %r — discarding the raw appointment_date; only the "
            "deterministic resolution of the phrase will be used.",
            parsed.get("appointment_date"), relative_phrase, state.user_message,
        )
        parsed["appointment_date"] = None

    resolved_relative_date = _resolve_relative_date(relative_phrase, today_date) if relative_phrase else None

    logger.info(
        "Date resolution: relative_phrase=%r today=%s -> resolved=%r (model's own raw appointment_date=%r)",
        relative_phrase, today_date.isoformat(), resolved_relative_date, parsed.get("appointment_date"),
        )
    if resolved_relative_date:
        resolved_appointment_date = resolved_relative_date
    else:
        resolved_appointment_date = _carry_forward(parsed, "appointment_date", state.appointment_date)

    if weekday_phrase and not has_explicit_date_signal:
        corrected_date = _resolve_relative_date(weekday_phrase, today_date)
        if corrected_date and corrected_date != resolved_appointment_date:
            logger.warning(
                "Weekday consistency check failed: resolved_appointment_date=%r does not "
                "match weekday phrase %r found in message %r. Overriding with deterministic "
                "resolution %r.",
                resolved_appointment_date, weekday_phrase, state.user_message, corrected_date,
            )
            resolved_appointment_date = corrected_date

    new_date_given_this_turn = bool(resolved_relative_date) or parsed.get("appointment_date") is not None
    if new_date_given_this_turn:
        resolved_appointment_time = parsed.get("appointment_time")
    else:
        resolved_appointment_time = _carry_forward(parsed, "appointment_time", state.appointment_time)

   
    resolved_recommended_doctors = (
        [] if resolved_doctor_name != state.doctor_name else state.recommended_doctors
    )

    result = {
        "intent": resolved_intent,
        "doctor_name": resolved_doctor_name,
        "appointment_date": resolved_appointment_date,
        "appointment_time": resolved_appointment_time,
        "duration_minutes": parsed.get("duration_minutes") or state.duration_minutes or 30,
        "patient_name": _carry_forward(parsed, "patient_name", state.patient_name),
        "phone_number": _carry_forward(parsed, "phone_number", state.phone_number),
        "reason_for_visit": _carry_forward(parsed, "reason_for_visit", state.reason_for_visit),
        "specialization_needed": _carry_forward(parsed, "specialization_needed", state.specialization_needed),
        "invitee_email": parsed.get("invitee_email") or state.invitee_email,
        "appointment_id": resolved_appointment_id,
        "suggested_alternative": resolved_suggested_alternative,
        "recommended_doctors": resolved_recommended_doctors,
        "rebook_requested": rebook_requested,
    }

    if parsed.get("intent") is None:
        fallback_response = parsed.get("fallback_response")
        if fallback_response:
            result["response_message"] = fallback_response

    return result