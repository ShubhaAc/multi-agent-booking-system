"""
Standalone diagnostic — read-only, never touches best_skill.md, never calls
the optimizer or the gate. Just runs the forward pass on train.jsonl against
the CURRENT best_skill.md and breaks down exactly where the failures are.

Usage (from repo root, same venv you already use for training):
    python -m skillopt.diagnose --skill supervisor

Drop this file at: skillopt/diagnose.py
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter

from skillopt import skill_store
from skillopt.config import TRAIN_SPLIT_PATH
from skillopt.rollout import load_dataset, run_batch
from skillopt.verifier import score_batch, FIELD_WEIGHTS, DEFAULT_WEIGHT, _ALL_FIELDS
async def diagnose(skill_id: str, train_path: str = TRAIN_SPLIT_PATH, field: str | None = None) -> None:
    skill_text = skill_store.load_best_skill(skill_id)
    dataset = load_dataset(train_path)

    triples = await run_batch(skill_text, dataset)
    scores = score_batch(triples)

    n = len(scores)
    strict_passed = sum(1 for s in scores if s.passed)
    mean_weighted = sum(s.weighted_score for s in scores) / n if n else 0.0

    print(f"\n=== Forward pass on {train_path} ({n} examples) ===")
    print(f"Strict pass count (all fields exact): {strict_passed}/{n}")
    print(f"Mean weighted score (what the gate actually compares): {mean_weighted:.4f}")
    print("NOTE: these two numbers can diverge — a patch can raise the weighted")
    print("score (partial credit) without moving the strict pass count at all.\n")

    # --- Per-field mismatch counts ---
    field_miss_counts: Counter[str] = Counter()
    for s in scores:
        for f in s.mismatched_fields():
            field_miss_counts[f] += 1

    print("=== Mismatches by field (sorted by frequency) ===")
    print(f"{'field':<22}{'misses':>8}{'miss %':>10}{'weight':>9}")
    for field_name in sorted(_ALL_FIELDS, key=lambda f: -field_miss_counts.get(f, 0)):
        misses = field_miss_counts.get(field_name, 0)
        if misses == 0 and field_name not in field_miss_counts:
            continue
        weight = FIELD_WEIGHTS.get(field_name, DEFAULT_WEIGHT)
        pct = 100 * misses / n if n else 0.0
        print(f"{field_name:<22}{misses:>8}{pct:>9.1f}%{weight:>9.1f}")

    # --- Intent confusion: unchanged ---
    intent_confusions: Counter[tuple] = Counter()
    for s in scores:
        if "intent" in s.mismatched_fields():
            pred = getattr(s.predicted, "intent")
            exp = getattr(s.expected, "intent")
            intent_confusions[(exp, pred)] += 1

    if intent_confusions:
        print("\n=== Intent confusion (expected -> predicted), sorted by count ===")
        print(f"{'expected':<18}{'predicted':<18}{'count':>7}")
        for (exp, pred), count in intent_confusions.most_common(20):
            print(f"{str(exp):<18}{str(pred):<18}{count:>7}")

    # --- Worked examples: --field overrides which field's failures to show;
    # falls back to the automatic top-miss field when not passed. ---
    if field is not None:
        if field not in _ALL_FIELDS:
            print(f"\n'--field {field}' is not a known SupervisorOutput field. Known fields: {sorted(_ALL_FIELDS)}")
            return
        target_field = field
    elif field_miss_counts:
        target_field, _ = field_miss_counts.most_common(1)[0]
    else:
        target_field = None

    if target_field:
        print(f"\n=== Sample failures on field: {target_field!r} (up to 5) ===")
        shown = 0
        for s in scores:
            if target_field in s.mismatched_fields():
                print(f"\n  user_message: {s.input_context.get('user_message')!r}")
                print(f"  prev_state:   {s.input_context.get('prev_state')}")
                print(f"  expected.{target_field} = {getattr(s.expected, target_field)!r}")
                print(f"  predicted.{target_field} = {getattr(s.predicted, target_field)!r}")
                shown += 1
                if shown >= 5:
                    break
        if shown == 0:
            print(f"  (no failures found for {target_field!r} — it's passing cleanly)")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m skillopt.diagnose")
    parser.add_argument("--skill", required=True, help="skill_id, e.g. 'supervisor'")
    parser.add_argument("--train-path", default=TRAIN_SPLIT_PATH)
    parser.add_argument("--field", default=None, help="inspect a specific field's failures instead of the automatic top-miss field, e.g. --field relative_date_phrase")
    args = parser.parse_args()
    asyncio.run(diagnose(args.skill, args.train_path, args.field))


if __name__ == "__main__":
    main()