"""
Forward pass: run agents.supervisor's structured-output LLM call against a
labeled dataset split, under a given candidate skill text, and return
(predicted, expected, input_context) tuples ready for verifier.score_batch.

Deliberately calls agents.supervisor.structured_llm directly instead of going
through supervisor_node(). supervisor_node() has two things training doesn't
want: the fast-path regex shortcuts (deterministic code, not skill text — a
skill patch can never change their behavior, so routing dataset items through
them would just test the regexes, not the skill) and side effects like DB
writes. Rollouts must be pure: same input + same skill text -> same call,
nothing else touched.

STATIC_INSTRUCTIONS is swapped for the candidate skill text for the duration
of the batch via a monkeypatch, then restored — this file never writes to
disk, that's skill_store's job. Batches run sequentially against the module
via a lock so concurrent rollouts against different candidate texts (e.g. an
epoch refactor running while a training step is also in flight) can't clobber
each other's STATIC_INSTRUCTIONS value mid-call.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date

import agents.supervisor as supervisor_module
from agents.supervisor import SupervisorOutput, DYNAMIC_CONTEXT_TEMPLATE
from langchain_core.messages import SystemMessage, HumanMessage

_PATCH_LOCK = asyncio.Lock()
_MAX_CONCURRENCY = 8  # bounds simultaneous OpenAI calls per batch


@dataclass
class DatasetItem:
    user_message: str
    prev_state: dict          # e.g. {"intent": None, "doctor_name": "Dr Lee", ...}
    expected_output: dict      # fields matching SupervisorOutput
    today: str | None = None   # YYYY-MM-DD, defaults to today if absent
    history: list[dict] | None = None  # [{"role": "user"|"assistant", "content": str}, ...]


def load_dataset(path: str) -> list[DatasetItem]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no} is not valid JSON") from e
            items.append(DatasetItem(
                user_message=raw["input"]["user_message"],
                prev_state=raw["input"].get("prev_state", {}),
                expected_output=raw["expected_output"],
                today=raw["input"].get("today"),
                history=raw["input"].get("history"),
            ))
    return items


def _build_dynamic_context(item: DatasetItem) -> str:
    today_date = date.fromisoformat(item.today) if item.today else date.today()
    history = item.history or []
    history_text = "".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}\n" for m in history
    ) or "No history yet."

    prev = item.prev_state
    return DYNAMIC_CONTEXT_TEMPLATE.format(
        today=today_date.isoformat(),
        today_weekday=today_date.strftime("%A"),
        history=history_text,
        prev_intent=prev.get("intent"),
        prev_doctor_name=prev.get("doctor_name"),
        prev_suggested_alternative=prev.get("suggested_alternative"),
        prev_appointment_id=prev.get("appointment_id"),
        prev_appointment_date=prev.get("appointment_date"),
        prev_appointment_time=prev.get("appointment_time"),
        prev_duration_minutes=prev.get("duration_minutes"),
        prev_patient_name=prev.get("patient_name"),
        prev_phone_number=prev.get("phone_number"),
        prev_reason_for_visit=prev.get("reason_for_visit"),
        prev_specialization_needed=prev.get("specialization_needed"),
        prev_invitee_email=prev.get("invitee_email"),
    )


@asynccontextmanager
async def _skill_applied(skill_text: str):

    async with _PATCH_LOCK:
        original = supervisor_module.STATIC_INSTRUCTIONS
        supervisor_module.STATIC_INSTRUCTIONS = skill_text
        try:
            yield
        finally:
            supervisor_module.STATIC_INSTRUCTIONS = original


async def _run_one(
    item: DatasetItem,
    semaphore: asyncio.Semaphore,
    callbacks=None,
) -> tuple[SupervisorOutput, SupervisorOutput, dict]:
    async with semaphore:
        dynamic_context = _build_dynamic_context(item)
        response = await supervisor_module.structured_llm.ainvoke(
            [
                SystemMessage(content=supervisor_module.STATIC_INSTRUCTIONS),
                SystemMessage(content=dynamic_context),
                HumanMessage(content=item.user_message),
            ],
            config={"callbacks": callbacks} if callbacks else None,
        )
        parsing_error = response.get("parsing_error")
        if parsing_error:
            raise ValueError(f"structured output failed to parse for input {item.user_message!r}: {parsing_error}")

        predicted: SupervisorOutput = response["parsed"]
        expected = SupervisorOutput(**item.expected_output)
        input_context = {
            "user_message": item.user_message,
            "prev_state": item.prev_state,
            "today": item.today,
        }
        return predicted, expected, input_context


async def run_batch(
    skill_text: str,
    dataset: list[DatasetItem],
    max_concurrency: int = _MAX_CONCURRENCY,
    callbacks=None,
) -> list[tuple[SupervisorOutput, SupervisorOutput, dict]]:
    semaphore = asyncio.Semaphore(max_concurrency)
    async with _skill_applied(skill_text):
        results = await asyncio.gather(*(_run_one(item, semaphore, callbacks) for item in dataset))
    return results


async def run_split(
    skill_text: str,
    dataset_path: str,
    max_concurrency: int = _MAX_CONCURRENCY,
    callbacks=None,
):
    dataset = load_dataset(dataset_path)
    return await run_batch(skill_text, dataset, max_concurrency=max_concurrency, callbacks=callbacks)