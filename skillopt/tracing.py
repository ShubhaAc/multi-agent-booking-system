
from __future__ import annotations

from opik.integrations.langchain import OpikTracer

from skillopt.config import SKILLOPT_OPIK_PROJECT_NAME


def make_tracer(skill_id: str, phase: str) -> OpikTracer:
    """phase: 'train-rollout' | 'val-baseline' | 'val-candidate' | 'optimizer'."""
    return OpikTracer(
        project_name=SKILLOPT_OPIK_PROJECT_NAME,
        tags=["skillopt", skill_id, phase],
        metadata={"skill_id": skill_id, "phase": phase},
    )