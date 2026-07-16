"""
One-off hotfix: remove intent/field lines from skills/supervisor/best_skill.md
that don't exist in agents.supervisor.SupervisorOutput's schema.

Run this from your repo root (where `skills/` and `agents/` live):
    python strip_schema_drift.py

It edits both best_skill.md (what's actually loaded at inference time) and
skill.md (the working copy, so the next training step doesn't just
reintroduce the same drift from the pre-hotfix working copy).
"""
import re

BEST_PATH = "skills/supervisor/best_skill.md"
WORKING_PATH = "skills/supervisor/skill.md"

# Exact bogus lines identified against SupervisorOutput's Literal enum.
DRIFTED_LINE_PATTERNS = [
    r"\n- clear_aligners_consultation:.*",
    r"\n- emergency_dental_issue:.*",
    r"\n- urgent: indicates the user needs an appointment.*",
]

def strip(text: str) -> tuple[str, int]:
    removed = 0
    for pattern in DRIFTED_LINE_PATTERNS:
        text, n = re.subn(pattern, "", text)
        removed += n
    return text, removed

def main():
    for path in (BEST_PATH, WORKING_PATH):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        cleaned, removed = strip(text)
        if removed == 0:
            print(f"{path}: no drifted lines found (already clean or path wrong)")
            continue
        with open(path, "w", encoding="utf-8") as f:
            f.write(cleaned)
        print(f"{path}: removed {removed} schema-drifted line(s)")

if __name__ == "__main__":
    main()