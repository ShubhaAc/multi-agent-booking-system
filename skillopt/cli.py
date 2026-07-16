

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
from skillopt import skill_store, rejected_buffer, validation_gate, review_queue
from skillopt.config import EDIT_BUDGET as DEFAULT_EDIT_BUDGET, TRAIN_SPLIT_PATH
from skillopt.optimizer import select_minibatch, propose_patches, to_skill_store_patches
from skillopt.rollout import load_dataset, run_batch
from skillopt.tracing import make_tracer
from skillopt.verifier import score_batch, split_by_label

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


async def run_training_step(skill_id: str, edit_budget: int, train_path: str = TRAIN_SPLIT_PATH) -> bool:
    
    base_text = skill_store.load_best_skill(skill_id)

    dataset = load_dataset(train_path)
    random.shuffle(dataset)  # vary which failures land in the minibatch across steps

    rollout_tracer = make_tracer(skill_id, "train-rollout")
    triples = await run_batch(base_text, dataset, callbacks=[rollout_tracer])
    rollout_tracer.flush()

    scores = score_batch(triples)
    successes, failures = split_by_label(scores)
    logger.info("Forward pass: %d/%d passed (%d failures)", len(successes), len(scores), len(failures))

    if not failures:
        logger.info("No failures in this pass — skill already fits the training split, nothing to patch.")
        return False

    minibatch = select_minibatch(successes, failures)
    buffer = rejected_buffer.load(skill_id)

    optimizer_tracer = make_tracer(skill_id, "optimizer")
    output = await propose_patches(base_text, minibatch, buffer, edit_budget, callbacks=[optimizer_tracer])
    optimizer_tracer.flush()

    if not output.patches:
        logger.info("Optimizer proposed no patches this cycle. gradient_summary=%r", output.gradient_summary)
        if output.deferred:
            for d in output.deferred:
                logger.info("Deferred: %s (%s)", d.issue, d.reason)
        return False

    logger.info("Optimizer proposed %d patch(es): %s", len(output.patches), output.gradient_summary)
    patches = to_skill_store_patches(output)

    try:
        candidate_text = skill_store.apply_patches(base_text, patches)
    except skill_store.PatchApplyError as e:
        logger.warning("Patch application failed, treating step as rejected: %s", e)
        rejected_buffer.add(skill_id, patches, reason=f"apply_error: {e}")
        return False

    skill_store.save_working_skill(skill_id, candidate_text)

    decision = await validation_gate.evaluate_and_gate(skill_id, candidate_text, patches)
    if decision.accepted:
        logger.info(
            "ACCEPTED -> v%d (baseline=%.4f, candidate=%.4f)",
            decision.committed_version, decision.baseline_score, decision.candidate_score,
        )
    else:
        logger.info("REJECTED (%s)", decision.reason)
    return decision.accepted


async def run_training(skill_id: str, steps: int, edit_budget: int) -> None:
    accepted_count = 0
    for step in range(1, steps + 1):
        logger.info("--- training step %d/%d ---", step, steps)
        accepted = await run_training_step(skill_id, edit_budget)
        accepted_count += int(accepted)
    logger.info("Training run complete: %d/%d steps accepted.", accepted_count, steps)


async def run_epoch(skill_id: str) -> None:
    
    from skillopt.epoch_refactor import run_epoch_refactor
    await run_epoch_refactor(skill_id)


async def run_sleep(skill_id: str) -> None:

    from skillopt.sleep import run_sleep_cycle
    await run_sleep_cycle(skill_id)


def _print_row_summary(entry: dict) -> None:
    inp = entry["input"]
    print(f"[{entry['id']}] status={entry['status']} flagged: {', '.join(entry['flag_reasons']) or '(none)'}")
    print(f"    user_message: {inp.get('user_message')!r}")
    print(f"    prev_state:   {inp.get('prev_state')}")
    print(f"    mined_expected_output: {entry['mined_expected_output']}")
    print()


def run_review(args: argparse.Namespace) -> None:
    skill_id = args.skill

    if args.list:
        pending = review_queue.list_pending(skill_id)
        if not pending:
            print(f"No pending rows for skill_id={skill_id!r}.")
            return
        print(f"{len(pending)} pending row(s) for skill_id={skill_id!r}:\n")
        for entry in pending:
            _print_row_summary(entry)
        return

    if args.show:
        entry = review_queue.get(skill_id, args.show)
        if entry is None:
            print(f"No row with id={args.show!r} under skill_id={skill_id!r}.")
            return
        _print_row_summary(entry)
        return

    if args.approve:
        corrected = json.loads(args.corrected) if args.corrected else None
        ok = review_queue.approve(skill_id, args.approve, corrected_expected_output=corrected)
        if ok:
            source = "corrected label" if corrected is not None else "mined (unverified) label as-is"
            print(f"Approved [{args.approve}] -> appended to train.jsonl using {source}.")
        else:
            print(f"Could not approve [{args.approve}] — not found or not pending.")
        return

    if args.reject:
        ok = review_queue.reject(skill_id, args.reject, reason=args.reason or "")
        if ok:
            print(f"Rejected [{args.reject}].")
        else:
            print(f"Could not reject [{args.reject}] — not found or not pending.")
        return

    print("Nothing to do — pass --list, --show, --approve, or --reject.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m skillopt.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Run N forward/backward/gate cycles.")
    train_p.add_argument("--skill", required=True, help="skill_id, e.g. 'supervisor'")
    train_p.add_argument("--steps", type=int, default=1)
    train_p.add_argument("--edit-budget", type=int, default=DEFAULT_EDIT_BUDGET)

    epoch_p = sub.add_parser("epoch", help="Run a macro-review/refactor pass over best_skill.md.")
    epoch_p.add_argument("--skill", required=True)

    sleep_p = sub.add_parser("sleep", help="Mine production traces into the pending review queue.")
    sleep_p.add_argument("--skill", required=True)

    review_p = sub.add_parser("review", help="Inspect and approve/reject rows mined by `sleep`.")
    review_p.add_argument("--skill", required=True)
    review_p.add_argument("--list", action="store_true", help="List all pending rows.")
    review_p.add_argument("--show", metavar="ID", help="Show one row in full.")
    review_p.add_argument("--approve", metavar="ID", help="Promote a row into train.jsonl.")
    review_p.add_argument(
        "--corrected", metavar="JSON",
        help="Corrected expected_output as a JSON object, used instead of the mined "
             "(unverified) output when approving. Omit to accept the mined output as-is.",
    )
    review_p.add_argument("--reject", metavar="ID", help="Discard a row without training on it.")
    review_p.add_argument("--reason", help="Reason for --reject (stored for audit).")

    rollback_p = sub.add_parser("rollback", help="Restore best_skill.md from a history snapshot.")
    rollback_p.add_argument("--skill", required=True)
    rollback_p.add_argument("--to-version", type=int, required=True)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "train":
        asyncio.run(run_training(args.skill, args.steps, args.edit_budget))
    elif args.command == "epoch":
        asyncio.run(run_epoch(args.skill))
    elif args.command == "sleep":
        asyncio.run(run_sleep(args.skill))
    elif args.command == "review":
        run_review(args)
    elif args.command == "rollback":
        skill_store.rollback(args.skill, args.to_version)
        logger.info("Rolled back %r to v%d", args.skill, args.to_version)


if __name__ == "__main__":
    main()