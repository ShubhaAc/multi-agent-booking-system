import json

path = "data/skillopt/val.jsonl"
rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

fixes = {
    "my son's braces checkup, whoever's available friday morning": "friday",
    "wisdom tooth is killing me, can dr okafor see me tomorrow morning": "tomorrow",
}

applied = 0
for r in rows:
    msg = r["input"]["user_message"]
    if msg in fixes:
        r["expected_output"]["relative_date_phrase"] = fixes[msg]
        applied += 1

assert applied == len(fixes), f"expected {len(fixes)}, got {applied}"
with open(path, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")