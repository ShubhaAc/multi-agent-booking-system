from __future__ import annotations

import logging
from collections import defaultdict

import opik

from config import OPIK_PROJECT_NAME
from skillopt import review_queue

logger = logging.getLogger(__name__)


def _fetch_recent_traces(limit: int = 200) -> list[dict]:
    client = opik.Opik()
    traces = client.get_traces(project_name=OPIK_PROJECT_NAME, max_results=limit)
    return traces


# Fields where "this turn's extracted value is literally a raw user message
# from earlier in the conversation" is a strong smell that the model echoed
# history instead of extracting from the current turn — this is exactly the
# shape of the patient_name bug this heuristic was written to catch. Kept to
# free-text identity fields; numeric/enum fields (duration_minutes, intent)
# have a much higher legitimate collision rate and would just generate noise.
_ECHO_CHECK_FIELDS = ("patient_name", "phone_number", "invitee_email", "doctor_name", "reason_for_visit")


def _echoed_history_fields(trace: dict) -> list[str]:
    """Returns which _ECHO_CHECK_FIELDS in this trace's output exactly match
    (case-insensitive, whitespace-stripped) some earlier raw user message —
    i.e. the model likely pulled a field from history instead of the current
    turn, regardless of what the current message actually said."""
    output = trace.get("output") or {}
    input_data = trace.get("input") or {}
    history = input_data.get("history") or []

    prior_user_messages = {
        (m.get("content") or "").strip().lower()
        for m in history
        if m.get("role") == "user"
    }
    prior_user_messages.discard("")

    hits = []
    for field in _ECHO_CHECK_FIELDS:
        value = output.get(field)
        if isinstance(value, str) and value.strip().lower() in prior_user_messages:
            hits.append(field)
    return hits


def _group_by_sender(traces: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for t in traces:
        sender_id = (t.get("input") or {}).get("prev_state", {}).get("sender_id") or t.get("metadata", {}).get("sender_id")
        if sender_id:
            groups[sender_id].append(t)
    return groups


def _stalled_field_traces(traces_for_sender: list[dict]) -> set[int]:
    """Within one sender's traces (assumed already in chronological order as
    returned by Opik), flags a trace index where a booking-critical field
    stayed exactly the same across two consecutive turns even though the
    user's message text changed — i.e. the user said something new and the
    extraction didn't move, suggesting the field got stuck rather than
    genuinely being unchanged. This is a weaker/noisier signal than the echo
    check above, so it only contributes a flag reason, never a standalone
    "definitely wrong" verdict."""
    stalled: set[int] = set()
    watch_fields = ("appointment_time", "appointment_date", "doctor_name")
    for i in range(1, len(traces_for_sender)):
        prev_t, cur_t = traces_for_sender[i - 1], traces_for_sender[i]
        prev_out, cur_out = prev_t.get("output") or {}, cur_t.get("output") or {}
        prev_msg = (prev_t.get("input") or {}).get("user_message", "")
        cur_msg = (cur_t.get("input") or {}).get("user_message", "")
        if not cur_msg or cur_msg.strip().lower() == prev_msg.strip().lower():
            continue
        for field in watch_fields:
            if cur_out.get(field) is None and prev_out.get(field) is None:
                continue
            # Only interesting when the field was MISSING and is still missing
            # after a new message — i.e. the flow appears stuck asking for it.
            if prev_out.get(field) is None and cur_out.get(field) is None:
                stalled.add(i)
    return stalled


def _looks_like_failure(trace: dict, echo_fields: list[str], stalled: bool) -> list[str]:
    """Returns the list of reasons this trace was flagged, or [] if it looks
    fine. Multiple independent signals rather than one boolean, so a human
    reviewer can see WHY something got mined without re-deriving it."""
    reasons = []
    output = trace.get("output") or {}

    if output.get("intent") is None:
        reasons.append("intent_null")
    if trace.get("metadata", {}).get("corrected_by_user"):
        reasons.append("corrected_by_user")
    if echo_fields:
        reasons.append(f"echoed_history:{','.join(echo_fields)}")
    if stalled:
        reasons.append("stalled_field_across_turns")

    return reasons


def _trace_to_dataset_row(trace: dict, flag_reasons: list[str]) -> dict | None:
    input_data = trace.get("input") or {}
    output_data = trace.get("output") or {}
    if not input_data.get("user_message") or not output_data:
        return None
    return {
        "input": {
            "user_message": input_data["user_message"],
            "prev_state": input_data.get("prev_state", {}),
            "today": input_data.get("today"),
            "history": input_data.get("history"),
        },
        # NOTE: this is the model's own production output, not a verified
        # gold label. review_queue.add() stores it as mined_expected_output
        # specifically so nothing downstream can mistake it for ground truth
        # until a human calls review_queue.approve() — see review_queue.py.
        "expected_output": output_data,
        "flag_reasons": flag_reasons,
    }


async def run_sleep_cycle(skill_id: str) -> None:
    """Mines recent Opik traces for likely supervisor mistakes and queues
    them for human review. Does NOT train automatically and does NOT touch
    train.jsonl directly — see review_queue.py for why. Run
    `python -m skillopt.cli review --skill <id> --list` afterwards to see
    what was queued, then approve/reject each row before it can affect
    training.
    """
    traces = _fetch_recent_traces()
    logger.info("Fetched %d recent traces from Opik project %r", len(traces), OPIK_PROJECT_NAME)

    sender_groups = _group_by_sender(traces)
    stalled_indices_by_sender = {
        sender: _stalled_field_traces(group) for sender, group in sender_groups.items()
    }

    rows = []
    for trace in traces:
        echo_fields = _echoed_history_fields(trace)

        sender_id = (trace.get("input") or {}).get("prev_state", {}).get("sender_id") or trace.get("metadata", {}).get("sender_id")
        stalled = False
        if sender_id and sender_id in sender_groups:
            group = sender_groups[sender_id]
            try:
                idx = group.index(trace)
                stalled = idx in stalled_indices_by_sender[sender_id]
            except ValueError:
                pass

        reasons = _looks_like_failure(trace, echo_fields, stalled)
        if not reasons:
            continue

        row = _trace_to_dataset_row(trace, reasons)
        if row is not None:
            rows.append(row)

    logger.info("Flagged %d/%d traces as likely failures", len(rows), len(traces))

    if not rows:
        logger.info("No usable rows mined from flagged traces — nothing queued for review.")
        return

    assigned_ids = review_queue.add(skill_id, rows)
    logger.warning(
        "%d rows queued for review under skill_id=%r (ids: %s). "
        "These are UNVERIFIED — nothing was written to train.jsonl and no "
        "training was triggered. Run "
        "`python -m skillopt.cli review --skill %s --list` to inspect them, "
        "then approve (optionally with a corrected label) or reject each one.",
        len(rows), skill_id, ", ".join(assigned_ids), skill_id,
    )