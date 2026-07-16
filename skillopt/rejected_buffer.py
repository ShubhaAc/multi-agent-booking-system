"""
Persists patches that were proposed by the optimizer but did not beat the
validation baseline (skillopt/validation_gate.py calls add() on every
rejection). optimizer.py reads this back on the next cycle and passes it to
the optimizer LLM as REJECTED_EDIT_BUFFER, so it doesn't keep re-proposing
the same losing edit — without this, a training loop with a noisy verifier
can oscillate forever between the same accept/reject pair.

Stored as one JSON file per skill_id under REJECTED_BUFFER_DIR, not a
database — this is small, append-mostly, human-inspectable data, and being
able to `cat` it during debugging is worth more here than query performance.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from skillopt.config import REJECTED_BUFFER_DIR
from skillopt.skill_store import Patch

# Oldest entries are dropped past this count (per skill_id) so the buffer —
# and therefore the optimizer prompt's REJECTED_EDIT_BUFFER section — doesn't
# grow without bound over months of training and start eating context budget
# for no benefit; a patch rejected 300 cycles ago is unlikely to be proposed
# again verbatim anyway.
MAX_ENTRIES_PER_SKILL = 200


def _path(skill_id: str) -> str:
    return os.path.join(REJECTED_BUFFER_DIR, f"{skill_id}.json")


def load(skill_id: str) -> list[dict]:
    """Returns [] for a skill that has never had a rejected patch — an
    empty/missing buffer is the normal starting state, not an error."""
    path = _path(skill_id)
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(skill_id: str, entries: list[dict]) -> None:
    os.makedirs(REJECTED_BUFFER_DIR, exist_ok=True)
    with open(_path(skill_id), "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def add(skill_id: str, patches: list[Patch], reason: str) -> None:
    """Appends one entry per patch in the rejected candidate — stored
    individually rather than grouped by candidate, since the optimizer
    reasons about REJECTED_EDIT_BUFFER at the level of "don't propose this
    specific bullet edit again," not "don't propose this exact combination
    of edits again." A future cycle might legitimately want half of a
    rejected candidate and not the other half."""
    entries = load(skill_id)
    timestamp = datetime.now(timezone.utc).isoformat()
    for patch in patches:
        entries.append({
            "op": patch.op,
            "section_id": patch.section_id,
            "target_text": patch.target_text,
            "new_text": patch.new_text,
            "reason": reason,
            "rejected_at": timestamp,
        })
    entries = entries[-MAX_ENTRIES_PER_SKILL:]
    _save(skill_id, entries)


def clear(skill_id: str) -> None:
    """Wipe the buffer for a skill. Intended for use after an epoch-level
    meta-refactor (skillopt/epoch_refactor.py) substantially rewrites the
    skill file — patches rejected against the pre-refactor wording may not
    even apply cleanly anymore, let alone be relevant, so starting the
    buffer fresh avoids carrying forward guidance that no longer makes
    sense. Not called automatically by anything else in this package."""
    _save(skill_id, [])