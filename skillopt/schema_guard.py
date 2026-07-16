from __future__ import annotations
import re
from typing import get_args
from agents.supervisor import SupervisorOutput

_INTENT_LITERALS = {v for v in get_args(SupervisorOutput.model_fields["intent"].annotation) if v is not None}
_INTENT_LITERALS.add("null")  # skill-file convention for representing the None/no-intent case
_ALL_FIELD_NAMES = set(SupervisorOutput.model_fields.keys())
_BULLET_TOKEN_RE = re.compile(r"^-\s*([a-z][a-z0-9_]*)\s*:", re.MULTILINE)


class SchemaDriftError(ValueError):
    pass


def check_intent_classification_section(section_text: str) -> None:
    for token in _BULLET_TOKEN_RE.findall(section_text):
        if token not in _INTENT_LITERALS:
            raise SchemaDriftError(
                f"{token!r} is not in SupervisorOutput.intent's Literal{sorted(_INTENT_LITERALS)}"
            )


def check_field_references(section_id: str, section_text: str) -> None:
    """Every '- token:' bullet outside the intent section must name an actual
    SupervisorOutput field. Catches hallucinated fields (e.g. an 'urgent'
    bullet when there is no such field) that the model has no schema slot to
    emit through — pure prompt noise that a bullet-count check wouldn't flag
    since each occurrence can be worded slightly differently."""
    for token in _BULLET_TOKEN_RE.findall(section_text):
        if token not in _ALL_FIELD_NAMES:
            raise SchemaDriftError(
                f"{token!r} in section {section_id!r} is not a field on SupervisorOutput "
                f"(known fields: {sorted(_ALL_FIELD_NAMES)})"
            )


def check_candidate(candidate_text: str) -> None:
    from skillopt.skill_store import parse_sections
    sections = parse_sections(candidate_text)
    if "intent" in sections:
        check_intent_classification_section(sections["intent"])
    for section_id, section_text in sections.items():
        if section_id != "intent":
            check_field_references(section_id, section_text)