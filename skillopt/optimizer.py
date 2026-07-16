"""
Backward pass: turn a batch of scored trajectories (verifier.TrajectoryScore)
into a bounded set of add/delete/replace patches against the current skill.

Uses structured output (same pattern as agents/supervisor.py's
SupervisorOutput) instead of asking the optimizer model for free-form JSON
and hand-parsing it — that removes an entire class of "the model wrapped its
JSON in a code fence" failures for free.

This module never writes to disk and never decides whether a patch gets kept
— it only proposes. skill_store.apply_patches() applies the proposal to get
a candidate text, and validation_gate.py decides whether that candidate
replaces best_skill.md.
"""

from __future__ import annotations

import json
import os
from typing import Literal, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from skillopt.config import OPTIMIZER_MODEL, MINIBATCH_SIZE
from skillopt.skill_store import Patch
from skillopt.verifier import TrajectoryScore

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "optimizer_prompt.md")


class PatchOut(BaseModel):
    op: Literal["add", "delete", "replace"]
    section_id: str = Field(description="The section to patch, e.g. 'intent', 'doctor_name', 'specialization', 'date_time', 'other_fields', 'tool_policy', 'verification', 'failure_recovery'.")
    target_text: Optional[str] = Field(default=None, description="Exact existing bullet text. Required for delete/replace.")
    new_text: Optional[str] = Field(default=None, description="Full new/replacement bullet text. Required for add/replace.")
    justification: str = Field(description="Cites trajectory indices this patch fixes, e.g. 'fixes #3, #7, #12'.")
    risk_note: str = Field(description="Which protected successes this could affect, or 'none'.")

    
class DeferredIssue(BaseModel):
    issue: str
    reason: str


class OptimizerOutput(BaseModel):
    gradient_summary: str = Field(description="1-3 sentences: what failure pattern this cycle's patches address.")
    patches: list[PatchOut] = Field(default_factory=list)
    deferred: list[DeferredIssue] = Field(default_factory=list)


def _load_optimizer_system_prompt() -> str:
    with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def select_minibatch(
    successes: list[TrajectoryScore],
    failures: list[TrajectoryScore],
    minibatch_size: int = MINIBATCH_SIZE,
) -> list[TrajectoryScore]:
    """Failures first (that's the actual gradient signal), successes fill
    the remainder so the optimizer has something to protect (Step 1 of the
    optimizer prompt). If there are more failures than minibatch_size, the
    excess is left for a later cycle rather than truncated arbitrarily —
    caller (skillopt/cli.py) is responsible for looping until failures run
    out or EDIT_BUDGET stops producing accepted patches."""
    batch = failures[:minibatch_size]
    remaining = minibatch_size - len(batch)
    if remaining > 0:
        batch += successes[:remaining]
    return batch


def _format_trajectory(index: int, t: TrajectoryScore) -> str:
    mismatches = t.mismatched_fields()
    mismatch_lines = "\n".join(
        f"    - {f}: predicted={getattr(t.predicted, f)!r} expected={getattr(t.expected, f)!r}"
        for f in mismatches
    ) or "    (none — all fields matched)"

    return (
        f"[{index}] label={t.label}\n"
        f"  user_message: {t.input_context.get('user_message')!r}\n"
        f"  prev_state: {t.input_context.get('prev_state')}\n"
        f"  mismatched_fields:\n{mismatch_lines}"
    )


def _format_minibatch(minibatch: list[TrajectoryScore]) -> str:
    return "\n\n".join(_format_trajectory(i, t) for i, t in enumerate(minibatch))


def _format_rejected_buffer(rejected_buffer: list[dict]) -> str:
    if not rejected_buffer:
        return "(empty — no previously rejected patches for this skill)"
    return "\n".join(
        f"- [{entry.get('op')}] section={entry.get('section_id')} "
        f"target={entry.get('target_text')!r} new={entry.get('new_text')!r} "
        f"(rejected: {entry.get('reason')})"
        for entry in rejected_buffer
    )


def _build_user_message(
    current_skill_text: str,
    minibatch: list[TrajectoryScore],
    rejected_buffer: list[dict],
    edit_budget: int,
) -> str:
    return (
        f"EDIT_BUDGET: {edit_budget}\n\n"
        f"CURRENT_SKILL:\n{current_skill_text}\n\n"
        f"MINIBATCH ({len(minibatch)} trajectories):\n{_format_minibatch(minibatch)}\n\n"
        f"REJECTED_EDIT_BUFFER:\n{_format_rejected_buffer(rejected_buffer)}"
    )


async def propose_patches(
    current_skill_text: str,
    minibatch: list[TrajectoryScore],
    rejected_buffer: list[dict],
    edit_budget: int,
    callbacks=None,
) -> OptimizerOutput:
    """Calls the optimizer LLM once and returns a validated OptimizerOutput.
    If MINIBATCH has no failures, short-circuits without a call — the
    optimizer prompt forbids cosmetic edits on an all-success batch, so
    there's nothing useful an API call could produce here."""
    if not any(t.label == "failure" for t in minibatch):
        return OptimizerOutput(gradient_summary="No failures in this minibatch — nothing to patch.")

    llm = ChatOpenAI(model=OPTIMIZER_MODEL, temperature=0, seed=42)
    structured_llm = llm.with_structured_output(OptimizerOutput, include_raw=True)

    system_prompt = _load_optimizer_system_prompt()
    user_message = _build_user_message(current_skill_text, minibatch, rejected_buffer, edit_budget)

    response = await structured_llm.ainvoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_message)],
        config={"callbacks": callbacks} if callbacks else None,
    )

    parsing_error = response.get("parsing_error")
    if parsing_error:
        raise ValueError(f"optimizer returned unparseable output: {parsing_error}")

    output: OptimizerOutput = response["parsed"]

    if len(output.patches) > edit_budget:
        raise ValueError(
            f"optimizer emitted {len(output.patches)} patches, exceeding EDIT_BUDGET={edit_budget} "
            f"— this is a hard constraint violation, not a soft warning."
        )

    return output


def to_skill_store_patches(output: OptimizerOutput) -> list[Patch]:
    """Convert the optimizer's PatchOut objects into skill_store.Patch
    objects ready for skill_store.apply_patches()."""
    return [
        Patch(
            op=p.op,
            section_id=p.section_id,
            target_text=p.target_text,
            new_text=p.new_text,
        )
        for p in output.patches
    ]