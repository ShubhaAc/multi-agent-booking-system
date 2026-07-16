import asyncio
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from skillopt import skill_store
from skillopt.rollout import load_dataset, run_batch
from skillopt.verifier import score_batch
from skillopt.config import VAL_SPLIT_PATH

FIELDS = ["clear_doctor", "reason_for_visit", "relative_date_phrase", "doctor_name", "fallback_response"]

async def main():
    skill_text = skill_store.load_best_skill("supervisor")
    dataset = load_dataset(VAL_SPLIT_PATH)
    print(f"Running {len(dataset)} val rows through the model...")
    triples = await run_batch(skill_text, dataset)
    print("Forward pass done, scoring...")
    scores = score_batch(triples)

    for field_name in FIELDS:
        print(f"\n{'='*60}\n{field_name}\n{'='*60}")
        shown = 0
        for s in scores:
            if field_name in s.mismatched_fields():
                print(f"\n  msg: {s.input_context.get('user_message')!r}")
                print(f"  prev_state: {s.input_context.get('prev_state')}")
                print(f"  expected.{field_name} = {getattr(s.expected, field_name)!r}")
                print(f"  predicted.{field_name} = {getattr(s.predicted, field_name)!r}")
                shown += 1
                if shown >= 6:
                    break

asyncio.run(main())