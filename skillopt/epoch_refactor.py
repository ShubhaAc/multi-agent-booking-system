from __future__ import annotations

import json
import logging
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from skillopt import skill_store, rejected_buffer, validation_gate
from skillopt.config import OPTIMIZER_MODEL
from skillopt.optimizer import OptimizerOutput, to_skill_store_patches

logger = logging.getLogger(__name__)

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "epoch_refactor_prompt.md")


def _load_epoch_prompt() -> str:
    with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


async def run_epoch_refactor(skill_id: str) -> None:
    current_text = skill_store.load_best_skill(skill_id)

    llm = ChatOpenAI(model=OPTIMIZER_MODEL, temperature=0)
    structured_llm = llm.with_structured_output(OptimizerOutput, include_raw=True)

    system_prompt = _load_epoch_prompt()
    user_message = f"CURRENT_SKILL:\n{current_text}"

    response = await structured_llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ])

    parsing_error = response.get("parsing_error")
    if parsing_error:
        raise ValueError(f"epoch refactor returned unparseable output: {parsing_error}")

    output: OptimizerOutput = response["parsed"]
    if not output.patches:
        logger.info("Epoch refactor proposed no changes. gradient_summary=%r", output.gradient_summary)
        return

    patches = to_skill_store_patches(output)
    candidate_text = skill_store.apply_patches(current_text, patches)
    skill_store.save_working_skill(skill_id, candidate_text)

    decision = await validation_gate.evaluate_and_gate(skill_id, candidate_text, patches)
    if decision.accepted:
        logger.info("Epoch refactor ACCEPTED -> v%d", decision.committed_version)
        rejected_buffer.clear(skill_id)
    else:
        logger.info("Epoch refactor REJECTED (%s)", decision.reason)