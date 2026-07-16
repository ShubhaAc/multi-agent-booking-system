from dataclasses import dataclass
from skillopt import skill_store
from skillopt.rejected_buffer import add as add_to_rejected_buffer
from skillopt.schema_guard import check_candidate, SchemaDriftError
from skillopt.rollout import run_split
from skillopt.verifier import aggregate, score_batch, TrajectoryScore
from skillopt.config import VAL_SPLIT_PATH, GATE_EVAL_PASSES, GATE_MIN_MARGIN
from skillopt.tracing import make_tracer

@dataclass
class EvaluationResult:
    aggregate_score: float
    scores: list[TrajectoryScore]


@dataclass
class GateDecision:
    accepted: bool
    baseline_score: float
    candidate_score: float
    committed_version: int | None   # set only if accepted
    reason: str


async def evaluate(
    skill_text: str,
    val_path: str = VAL_SPLIT_PATH,
    callbacks=None,
) -> EvaluationResult:
  
    triples = await run_split(skill_text, val_path, callbacks=callbacks)
    scores = score_batch(triples)
    return EvaluationResult(aggregate_score=aggregate(scores), scores=scores)



async def evaluate_and_gate(
    skill_id: str,
    candidate_text: str,
    patches: list[skill_store.Patch],
    val_path: str = VAL_SPLIT_PATH,
) -> GateDecision:
    baseline_text = skill_store.load_best_skill(skill_id)

    try:
        check_candidate(candidate_text)
    except SchemaDriftError as e:
        add_to_rejected_buffer(skill_id, patches, reason=f"schema drift: {e}")
        return GateDecision(
            accepted=False, baseline_score=0.0, candidate_score=0.0,
            committed_version=None, reason=f"schema drift: {e}",
        )


    baseline_scores: list[float] = []
    candidate_scores: list[float] = []
    for i in range(GATE_EVAL_PASSES):
        baseline_tracer = make_tracer(skill_id, f"val-baseline-pass{i}")
        candidate_tracer = make_tracer(skill_id, f"val-candidate-pass{i}")
        baseline_result = await evaluate(baseline_text, val_path, callbacks=[baseline_tracer])
        candidate_result = await evaluate(candidate_text, val_path, callbacks=[candidate_tracer])
        baseline_tracer.flush()
        candidate_tracer.flush()
        baseline_scores.append(baseline_result.aggregate_score)
        candidate_scores.append(candidate_result.aggregate_score)

    baseline_mean = sum(baseline_scores) / len(baseline_scores)
    candidate_mean = sum(candidate_scores) / len(candidate_scores)

    if candidate_mean > baseline_mean + GATE_MIN_MARGIN:
        version = skill_store.commit(skill_id, candidate_text)
        return GateDecision(
            accepted=True,
            baseline_score=baseline_mean,
            candidate_score=candidate_mean,
            committed_version=version,
            reason=(
                f"candidate scored {candidate_mean:.4f} (passes={candidate_scores}) "
                f"> baseline {baseline_mean:.4f} (passes={baseline_scores}) + margin {GATE_MIN_MARGIN}"
            ),
        )

    reason = (
        f"candidate scored {candidate_mean:.4f} (passes={candidate_scores}) "
        f"did not clear baseline {baseline_mean:.4f} (passes={baseline_scores}) + margin {GATE_MIN_MARGIN}"
    )
    add_to_rejected_buffer(skill_id, patches, reason=reason)
    return GateDecision(
        accepted=False,
        baseline_score=baseline_mean,
        candidate_score=candidate_mean,
        committed_version=None,
        reason=reason,
    )