
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