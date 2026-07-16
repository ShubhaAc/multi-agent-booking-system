import os
from config import MODEL_NAME, OPIK_PROJECT_NAME

OPTIMIZER_MODEL = os.getenv("SKILLOPT_OPTIMIZER_MODEL", MODEL_NAME)

EDIT_BUDGET = int(os.getenv("SKILLOPT_EDIT_BUDGET", "4"))

EPOCH_SIZE = int(os.getenv("SKILLOPT_EPOCH_SIZE", "10"))

MINIBATCH_SIZE = int(os.getenv("SKILLOPT_MINIBATCH_SIZE", "40"))

TRAIN_SPLIT_PATH = os.getenv("SKILLOPT_TRAIN_SPLIT_PATH", "data/skillopt/train.jsonl")
VAL_SPLIT_PATH = os.getenv("SKILLOPT_VAL_SPLIT_PATH", "data/skillopt/val.jsonl")

SKILLS_DIR = os.getenv("SKILLOPT_SKILLS_DIR", "skills")

REJECTED_BUFFER_DIR = os.getenv("SKILLOPT_REJECTED_BUFFER_DIR", "skillopt/.rejected_buffer")

PENDING_REVIEW_DIR = os.getenv("SKILLOPT_PENDING_REVIEW_DIR", "skillopt/.pending_review")

SKILLOPT_OPIK_PROJECT_NAME = os.getenv("SKILLOPT_OPIK_PROJECT_NAME", f"{OPIK_PROJECT_NAME}-skillopt")

GATE_EVAL_PASSES = int(os.getenv("SKILLOPT_GATE_EVAL_PASSES", "2"))
GATE_MIN_MARGIN = float(os.getenv("SKILLOPT_GATE_MIN_MARGIN", "0.005"))