

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from skillopt.config import REJECTED_BUFFER_DIR
from skillopt.skill_store import Patch


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
  
    _save(skill_id, [])

    