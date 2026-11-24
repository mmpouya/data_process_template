"""Markdown skills for tau agent harnesses (core layer).

Inspired by tau's coding layer (`tau_coding/skills.py`) but kept minimal and
independent of `tau_coding`, so it only depends on `tau_agent` and PyYAML.

A skill follows the Agent Skills layout: a directory containing a `SKILL.md`
file, optionally starting with a YAML frontmatter block:

    my-skill/
      SKILL.md      <- "---\ndescription: ...\n---" + markdown body
      notes.md      <- optional reference files, mentioned by SKILL.md

Skills reach the agent in two ways (see `docs/skills.md`):

1. The system prompt lists available skills (`build_skill_index`).
2. The agent loads one on demand through the `use_skill` tool
   (`create_use_skill_tool`), which returns the full skill body.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import yaml
from pydantic import BaseModel

from .exceptions import AgentSkillError

DEFAULT_SKILLS_DIR = Path(__file__).parent / "skills"


class Skill(BaseModel):
    """A markdown skill resource."""

    name: str
    path: Path
    content: str
    description: str | None = None


def load_skills(skills_dir: Path | str) -> list[Skill]:
    """Load skills from a directory.

    Every immediate subdirectory containing a ``SKILL.md`` file is one skill;
    the directory name is the skill name. Bare ``*.md`` files and directories
    without a ``SKILL.md`` are ignored. A missing directory yields ``[]``.
    """
    root = Path(skills_dir)
    if not root.is_dir():
        return []

    skills: list[Skill] = []
    for entry in sorted(root.iterdir(), key=lambda path: path.name):
        skill_file = entry / "SKILL.md"
        if not entry.is_dir() or not skill_file.is_file():
            continue
        raw = skill_file.read_text(encoding="utf-8")
        metadata, content = _split_frontmatter(raw)
        description = metadata.get("description") or _derive_description(content)
        skills.append(
            Skill(
                name=entry.name,
                path=skill_file,
                content=content.strip(),
                description=description,
            )
        )
    return skills


def find_skill(skills: Sequence[Skill], name: str) -> Skill:
    """Return the skill with the given name or raise ``AgentSkillError``."""
    for skill in skills:
        if skill.name == name:
            return skill
    available = ", ".join(sorted(skill.name for skill in skills)) or "none"
    raise AgentSkillError(f"Unknown skill: {name}. Available skills: {available}")


def build_skill_index(skills: Sequence[Skill]) -> str:
    """Build a concise skill list for system prompts and tool descriptions."""
    if not skills:
        return "Available skills: none"
    lines = ["Available skills:"]
    for skill in sorted(skills, key=lambda item: item.name):
        lines.append(f"- {skill.name}: {skill.description or 'No description'}")
    return "\n".join(lines)


def format_skill(skill: Skill) -> str:
    """Format a skill body the way tau injects skills into the transcript."""
    return (
        f'<skill name="{skill.name}" location="{skill.path}">\n'
        f"References are relative to {skill.path.parent}.\n\n"
        f"{skill.content.strip()}\n"
        "</skill>"
    )


def _split_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    """Split an optional leading YAML frontmatter block from markdown."""
    if not raw.startswith("---\n"):
        return {}, raw
    closing = raw.find("\n---", 4)
    if closing == -1:
        return {}, raw
    header = raw[4:closing]
    body = raw[closing + len("\n---") :].lstrip("\n")
    try:
        metadata = yaml.safe_load(header)
    except yaml.YAMLError as exc:
        raise AgentSkillError(f"Invalid skill frontmatter: {exc}") from exc
    if not isinstance(metadata, dict):
        return {}, raw
    return {str(key): str(value) for key, value in metadata.items()}, body


def _derive_description(content: str) -> str | None:
    """Use the first non-heading, non-empty line as a fallback description."""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    return None
