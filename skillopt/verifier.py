"""
Score a supervisor prediction against a gold label.

Deliberately NOT an LLM-judge. SupervisorOutput is a typed Pydantic model,
so "did the agent get it right" is a deterministic field comparison, not a
matter of opinion. That makes this verifier fast, free, and reproducible —
the same (predicted, expected) pair always scores the same way, which is
required for the validation gate's "strictly higher" comparison in
validation_gate.py to mean anything.

Field weighting matters here because not all fields are equally load-bearing:
intent alone decides which node graph.py routes to next (route_intent), and
doctor_name feeds directly into availability lookups and the eventual booking
record. A miss on either is a routing/booking failure, not just a metadata
miss — so both are weighted well above the remaining fields when computing
the aggregate score used by the validation gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from agents.supervisor import SupervisorOutput

# Fields that determine routing or write directly to a booking record are
# weighted higher than incidental metadata. Anything not listed here falls
# back to DEFAULT_WEIGHT. Tune these based on what SkillOpt-Sleep mining
# (skillopt/sleep.py) finds actually causes production incidents.
FIELD_WEIGHTS: dict[str, float] = {
    "intent": 3.0,
    "doctor_name": 2.0,
    "appointment_id": 2.0,
    "clear_doctor": 1.5,
    "rebook_requested": 1.5,
}
DEFAULT_WEIGHT = 1.0

_ALL_FIELDS = tuple(SupervisorOutput.model_fields.keys())


@dataclass
class TrajectoryScore:
    """Per-trajectory verifier result. `predicted` / `expected` /
    `input_context` are carried through unchanged so the optimizer prompt
    (skillopt/optimizer.py) can cite exact trajectories in its patch
    justifications instead of just an index number."""

    input_context: dict
    predicted: SupervisorOutput
    expected: SupervisorOutput
    field_diffs: dict[str, bool] = field(default_factory=dict)   # True = mismatch
    weighted_score: float = 0.0                                   # 0.0-1.0, 1.0 = perfect
    passed: bool = False
    label: Literal["success", "failure"] = "failure"

    def mismatched_fields(self) -> list[str]:
        return [f for f, is_diff in self.field_diffs.items() if is_diff]


def _normalize(field_name: str, value):
    if field_name == "fallback_response" and value == "":
        return None
    return value

def score(predicted: SupervisorOutput, expected: SupervisorOutput, input_context: dict) -> TrajectoryScore:


    field_diffs = {
        f: (_normalize(f, getattr(predicted, f)) != _normalize(f, getattr(expected, f))) for f in _ALL_FIELDS
    }
    total_weight = sum(FIELD_WEIGHTS.get(f, DEFAULT_WEIGHT) for f in _ALL_FIELDS)
    lost_weight = sum(
        FIELD_WEIGHTS.get(f, DEFAULT_WEIGHT) for f, is_diff in field_diffs.items() if is_diff
    )
    weighted_score = 1.0 - (lost_weight / total_weight)

    passed = not any(field_diffs.values())

    return TrajectoryScore(
        input_context=input_context,
        predicted=predicted,
        expected=expected,
        field_diffs=field_diffs,
        weighted_score=weighted_score,
        passed=passed,
        label="success" if passed else "failure",
    )


def score_batch(
    pairs: list[tuple[SupervisorOutput, SupervisorOutput, dict]],
) -> list[TrajectoryScore]:
    """pairs is [(predicted, expected, input_context), ...] — the shape
    rollout.py's run_batch() returns after it's done calling the agent."""
    return [score(p, e, ctx) for p, e, ctx in pairs]


def aggregate(scores: list[TrajectoryScore]) -> float:
    """Mean weighted_score across a batch — this is the single number
    validation_gate.py compares (candidate vs current best) to decide
    whether a training step is accepted. Empty batch scores 0.0 rather than
    raising, so a misconfigured/empty val split fails closed instead of
    silently passing every candidate."""
    if not scores:
        return 0.0
    return sum(s.weighted_score for s in scores) / len(scores)


def split_by_label(scores: list[TrajectoryScore]) -> tuple[list[TrajectoryScore], list[TrajectoryScore]]:
    """Returns (successes, failures) — the exact split optimizer.py needs
    before it builds a minibatch, per Step 1 of the optimizer prompt."""
    successes = [s for s in scores if s.label == "success"]
    failures = [s for s in scores if s.label == "failure"]
    return successes, failures