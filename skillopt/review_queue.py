"""
Holding area between skillopt/sleep.py (which mines candidate training rows
out of production traces) and data/skillopt/train.jsonl (which the training
loop treats as gold-labeled ground truth).

The old sleep.py wrote mined rows straight into train.jsonl using the
supervisor's OWN production output as expected_output — i.e. training the
skill to keep producing whatever it already produced, bugs included. This
module exists so that never happens again: mined rows always land here
first, and only move into train.jsonl via approve(), which requires an
explicit (possibly corrected) expected_output rather than trusting the
mined one blindly.

Storage: one JSON file per skill_id, same pattern as rejected_buffer.py —
small, append-mostly, human-inspectable, worth more here than a database.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone

from skillopt.config import PENDING_REVIEW_DIR, TRAIN_SPLIT_PATH


def _path(skill_id: str) -> str:
    return os.path.join(PENDING_REVIEW_DIR, f"{skill_id}.json")


def load(skill_id: str) -> list[dict]:
    """Returns [] for a skill with nothing pending — normal starting state."""
    path = _path(skill_id)
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(skill_id: str, entries: list[dict]) -> None:
    os.makedirs(PENDING_REVIEW_DIR, exist_ok=True)
    with open(_path(skill_id), "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def add(skill_id: str, rows: list[dict]) -> list[str]:
    """Appends mined rows to the pending queue. Each row must already have
    the shape {"input": {...}, "expected_output": {...}}. expected_output at
    this point is the supervisor's raw production output — labeled clearly
    as UNVERIFIED until a human/judge approves or corrects it.

    Returns the list of ids assigned, so callers (sleep.py) can log them.
    """
    entries = load(skill_id)
    now = datetime.now(timezone.utc).isoformat()
    assigned_ids = []
    for row in rows:
        row_id = uuid.uuid4().hex[:8]
        assigned_ids.append(row_id)
        entries.append({
            "id": row_id,
            "input": row["input"],
            "mined_expected_output": row["expected_output"],  # UNVERIFIED — model's own output
            "flag_reasons": row.get("flag_reasons", []),
            "status": "pending",
            "mined_at": now,
        })
    _save(skill_id, entries)
    return assigned_ids


def get(skill_id: str, row_id: str) -> dict | None:
    for entry in load(skill_id):
        if entry["id"] == row_id:
            return entry
    return None


def list_pending(skill_id: str) -> list[dict]:
    return [e for e in load(skill_id) if e["status"] == "pending"]


def approve(
    skill_id: str,
    row_id: str,
    corrected_expected_output: dict | None = None,
    train_path: str = TRAIN_SPLIT_PATH,
) -> bool:
    """Promotes a pending row into train.jsonl as a real gold-labeled
    example. If corrected_expected_output is given, that's what's written —
    NOT the mined (production) output. If omitted, the mined output is used
    as-is, but the caller is explicitly choosing that by omission; there is
    no silent default path that gets here without a human calling approve().

    Returns False if row_id isn't found or isn't still pending.
    """
    entries = load(skill_id)
    target = None
    for entry in entries:
        if entry["id"] == row_id:
            target = entry
            break
    if target is None or target["status"] != "pending":
        return False

    expected_output = corrected_expected_output if corrected_expected_output is not None else target["mined_expected_output"]

    train_row = {
        "input": target["input"],
        "expected_output": expected_output,
    }
    with open(train_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(train_row) + "\n")

    target["status"] = "approved"
    target["approved_at"] = datetime.now(timezone.utc).isoformat()
    target["final_expected_output"] = expected_output
    _save(skill_id, entries)
    return True


def reject(skill_id: str, row_id: str, reason: str = "") -> bool:
    """Marks a pending row rejected without promoting it. Kept in the file
    (not deleted) so the review history is auditable — same reasoning as
    rejected_buffer.py keeping losing patches around."""
    entries = load(skill_id)
    target = None
    for entry in entries:
        if entry["id"] == row_id:
            target = entry
            break
    if target is None or target["status"] != "pending":
        return False

    target["status"] = "rejected"
    target["rejected_at"] = datetime.now(timezone.utc).isoformat()
    target["rejection_reason"] = reason
    _save(skill_id, entries)
    return True