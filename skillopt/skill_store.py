

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from typing import Literal

from skillopt.config import SKILLS_DIR

_META_RE = re.compile(r"<!--\s*SKILLOPT-META(.*?)-->", re.DOTALL)
_META_FIELD_RE = re.compile(r"^\s*(\w+):\s*(.*?)\s*$", re.MULTILINE)

# Matches a section boundary: "## <!-- id:SECTION_ID -->" on its own line.
_SECTION_MARKER_RE = re.compile(r"^## <!--\s*id:([a-zA-Z0-9_]+)\s*-->\s*$", re.MULTILINE)


@dataclass
class Patch:
    op: Literal["add", "delete", "replace"]
    section_id: str
    target_text: str | None = None   # required for delete/replace, exact existing bullet
    new_text: str | None = None      # required for add/replace, full new bullet text


class SkillNotFoundError(FileNotFoundError):
    pass


class PatchApplyError(ValueError):
    """Raised when a patch's target_text doesn't match anything in the named
    section — either verbatim or whitespace-normalized. Almost always means
    the patch was computed against a stale copy of the skill (another patch
    already changed that bullet), or matched more than one location."""


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

def _skill_dir(skill_id: str) -> str:
    return os.path.join(SKILLS_DIR, skill_id)


def _skill_path(skill_id: str) -> str:
    return os.path.join(_skill_dir(skill_id), "skill.md")


def _best_path(skill_id: str) -> str:
    return os.path.join(_skill_dir(skill_id), "best_skill.md")


def _history_dir(skill_id: str) -> str:
    return os.path.join(_skill_dir(skill_id), "history")


def _history_path(skill_id: str, version: int) -> str:
    return os.path.join(_history_dir(skill_id), f"v{version}.md")


def _read(path: str) -> str:
    if not os.path.isfile(path):
        raise SkillNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# --------------------------------------------------------------------------
# Meta header (skill_id / version / parent / schema)
# --------------------------------------------------------------------------

def get_meta(raw_text: str) -> dict:
    match = _META_RE.search(raw_text)
    if not match:
        return {}
    return dict(_META_FIELD_RE.findall(match.group(1)))


def set_meta_version(raw_text: str, new_version: int) -> str:
    match = _META_RE.search(raw_text)
    if not match:
        return raw_text
    block = match.group(0)
    updated_block = re.sub(r"^version:\s*\d+\s*$", f"version: {new_version}", block, flags=re.MULTILINE)
    return raw_text[:match.start()] + updated_block + raw_text[match.end():]


# --------------------------------------------------------------------------
# Section parsing / patching
# --------------------------------------------------------------------------

def parse_sections(raw_text: str) -> dict[str, str]:
    """Split raw_text into {section_id: full_section_text}, where
    full_section_text includes the marker line and heading line, up to (not
    including) the next marker. Content before the first marker (the meta
    block + title) is stored under the key "__preamble__"."""
    markers = list(_SECTION_MARKER_RE.finditer(raw_text))
    if not markers:
        return {"__preamble__": raw_text}

    sections = {"__preamble__": raw_text[: markers[0].start()]}
    for i, m in enumerate(markers):
        section_id = m.group(1)
        start = m.start()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(raw_text)
        sections[section_id] = raw_text[start:end]
    return sections


def render_sections(sections: dict[str, str]) -> str:
    """Inverse of parse_sections — preamble first, then sections in insertion order."""
    ordered = [sections["__preamble__"]] + [
        text for sid, text in sections.items() if sid != "__preamble__"
    ]
    return "".join(ordered)


def _fuzzy_pattern(target_text: str) -> re.Pattern:
    """Build a regex that matches target_text but treats any run of
    whitespace in it as "any run of whitespace" in the source — so a patch
    generated against slightly different line-wrapping/indentation than
    what's actually on disk still finds its target. Everything that isn't
    whitespace is matched literally (re.escape'd)."""
    tokens = re.split(r"(\s+)", target_text.strip())
    parts = [r"\s+" if tok.isspace() else re.escape(tok) for tok in tokens if tok]
    return re.compile("".join(parts), re.DOTALL)


def _find_target_span(section_text: str, target_text: str) -> str | None:
    """Resolve target_text to the exact substring actually present in
    section_text. Tries a literal substring match first (fast path, covers
    the common case unchanged); falls back to a whitespace-tolerant regex
    only if that fails. Returns None (never guesses) if the fuzzy match
    finds zero or more-than-one occurrence, since applying to the wrong of
    two ambiguous matches is worse than failing the patch."""
    if not target_text:
        return None
    if target_text in section_text:
        return target_text
    matches = list(_fuzzy_pattern(target_text).finditer(section_text))
    if len(matches) == 1:
        return matches[0].group(0)
    return None


_BULLET_TOKEN_RE = re.compile(r"^-\s*([a-z][a-z0-9_]*)\s*:")

def _bullet_token(line: str) -> str | None:
    m = _BULLET_TOKEN_RE.match(line.strip())
    return m.group(1) if m else None


def apply_patch(raw_text: str, patch: Patch) -> str:
    sections = parse_sections(raw_text)

    if patch.op == "add":
        if patch.section_id not in sections:
            raise PatchApplyError(f"add: unknown section_id {patch.section_id!r}")
        if not patch.new_text:
            raise PatchApplyError("add: new_text is required")
        new_token = _bullet_token(patch.new_text)
        if new_token is not None:
            existing_tokens = {_bullet_token(line) for line in sections[patch.section_id].splitlines()}
            if new_token in existing_tokens:
                raise PatchApplyError(
                    f"add: section {patch.section_id!r} already has a {new_token!r} bullet — "
                    f"use 'replace' to edit it instead of adding a duplicate"
                )
        section = sections[patch.section_id].rstrip("\n")
        sections[patch.section_id] = f"{section}\n{patch.new_text.strip()}\n\n"


    elif patch.op == "delete":
        if patch.section_id not in sections:
            raise PatchApplyError(f"delete: unknown section_id {patch.section_id!r}")
        span_text = _find_target_span(sections[patch.section_id], patch.target_text or "")
        if span_text is None:
            raise PatchApplyError(
                f"delete: target_text not found (verbatim or whitespace-normalized) in "
                f"section {patch.section_id!r} — either the patch was computed against a "
                f"stale skill version, or it matched more than one location"
            )
        sections[patch.section_id] = sections[patch.section_id].replace(span_text + "\n", "", 1)

    elif patch.op == "replace":
        if patch.section_id not in sections:
            raise PatchApplyError(f"replace: unknown section_id {patch.section_id!r}")
        span_text = _find_target_span(sections[patch.section_id], patch.target_text or "")
        if span_text is None:
            raise PatchApplyError(
                f"replace: target_text not found (verbatim or whitespace-normalized) in "
                f"section {patch.section_id!r} — either the patch was computed against a "
                f"stale skill version, or it matched more than one location"
            )
        if not patch.new_text:
            raise PatchApplyError("replace: new_text is required")
        sections[patch.section_id] = sections[patch.section_id].replace(
            span_text, patch.new_text.strip(), 1
        )

    else:
        raise PatchApplyError(f"unknown op {patch.op!r}")

    return render_sections(sections)


def apply_patches(raw_text: str, patches: list[Patch]) -> str:
    text = raw_text
    applied = 0
    failures: list[str] = []
    for patch in patches:
        try:
            text = apply_patch(text, patch)
            applied += 1
        except PatchApplyError as e:
            failures.append(f"[{patch.op} {patch.section_id!r}] {e}")
    if applied == 0:
        detail = "; ".join(failures) if failures else "no patches given"
        raise PatchApplyError(f"all patches failed to apply: {detail}")
    return text


def load_working_skill(skill_id: str) -> str:
    """Raw skill.md, anchors intact — this is what training reads and patches."""
    return _read(_skill_path(skill_id))


def load_best_skill(skill_id: str) -> str:
    """Raw best_skill.md, anchors intact — the last accepted version."""
    return _read(_best_path(skill_id))


def load_active_skill(skill_id: str) -> str:
    """Plain instruction text for the running agent: strips the SKILLOPT-META
    comment and every `## <!-- id:X -->` marker line, leaving only the
    human-readable headings and rules. This is what gets assigned to
    STATIC_INSTRUCTIONS in agents/supervisor.py — the marker syntax is
    training-time bookkeeping the LLM has no reason to see at inference time."""
    raw = load_best_skill(skill_id)
    text = _META_RE.sub("", raw)
    text = _SECTION_MARKER_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
    return text


# --------------------------------------------------------------------------
# Public write API
# --------------------------------------------------------------------------

def save_working_skill(skill_id: str, raw_text: str) -> None:
    """Overwrite skill.md with a new candidate — used mid-training, before the
    candidate has cleared the validation gate. Does NOT touch best_skill.md."""
    _write(_skill_path(skill_id), raw_text)


def next_version(skill_id: str) -> int:
    history_dir = _history_dir(skill_id)
    if not os.path.isdir(history_dir):
        return 1
    existing = [
        int(m.group(1))
        for fname in os.listdir(history_dir)
        if (m := re.match(r"v(\d+)\.md$", fname))
    ]
    return (max(existing) + 1) if existing else 1


def commit(skill_id: str, candidate_text: str) -> int:
    """Called only after validation_gate.evaluate() confirms candidate_text
    scores strictly higher than the current best_skill.md. Writes the new
    best_skill.md, snapshots it into history/, and resets skill.md to match
    (so the next training step starts from the accepted state, not from
    whatever unaccepted edits were sitting in skill.md before)."""
    version = next_version(skill_id)
    versioned_text = set_meta_version(candidate_text, version)

    _write(_best_path(skill_id), versioned_text)
    _write(_history_path(skill_id, version), versioned_text)
    save_working_skill(skill_id, versioned_text)

    return version


def rollback(skill_id: str, to_version: int) -> None:
    """Restore best_skill.md (and reset skill.md to match) from a history
    snapshot. Does not delete history for versions after to_version — if a
    later training run creates version N+1 again, next_version() will just
    reuse numbers already present in history; that's fine, they'll be
    overwritten with fresh content."""
    snapshot = _read(_history_path(skill_id, to_version))
    _write(_best_path(skill_id), snapshot)
    save_working_skill(skill_id, snapshot)


def init_skill(skill_id: str, initial_text: str) -> None:
    """One-time setup for a brand new skill: writes skill.md and best_skill.md
    as identical copies of initial_text, and creates the (empty) history dir.
    Raises if the skill already exists, so this can't accidentally clobber
    an in-progress training run."""
    if os.path.isfile(_skill_path(skill_id)) or os.path.isfile(_best_path(skill_id)):
        raise FileExistsError(f"skill {skill_id!r} already initialized in {_skill_dir(skill_id)}")
    _write(_skill_path(skill_id), initial_text)
    _write(_best_path(skill_id), initial_text)
    os.makedirs(_history_dir(skill_id), exist_ok=True)