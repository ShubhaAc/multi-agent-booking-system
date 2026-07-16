import json, random

random.seed(42)

train_path = "data/skillopt/train.jsonl"
val_path = "data/skillopt/val.jsonl"

def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

train = load(train_path)
val = load(val_path)

print("before -> train:", len(train), "val:", len(val))

random.shuffle(train)
move_count = 45
moved = train[:move_count]
remaining_train = train[move_count:]

val_new = val + moved

with open(train_path, "w", encoding="utf-8") as f:
    for row in remaining_train:
        f.write(json.dumps(row) + "\n")

with open(val_path, "w", encoding="utf-8") as f:
    for row in val_new:
        f.write(json.dumps(row) + "\n")

print("after  -> train:", len(remaining_train), "val:", len(val_new))