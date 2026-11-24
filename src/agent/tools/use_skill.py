"""Skill loading tool (`use_skill`) for the tau agent harness.

Moved from ``src/agent/tools/authors/skill_use.py`` into the orchestration
package: it is an agent-level capability, not a catalog scraper. The tool
description embeds the skill index so the model always sees what is
available without paying the token cost of every skill body.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from tau_agent.messages import TextContent
from tau_agent.tools import AgentTool, AgentToolResult

from ..skill import Skill, build_skill_index, format_skill


def create_use_skill_tool(skills: Sequence[Skill]) -> AgentTool:
    """Create the ``use_skill`` tool that loads a skill body on demand."""
    skills_by_name = {skill.name: skill for skill in skills}

    async def executor(
        tool_call_id, arguments: Mapping[str, object], signal=None, on_update=None
    ) -> AgentToolResult:
        del tool_call_id, signal, on_update
        name = str(arguments.get("name", ""))
        skill = skills_by_name.get(name)
        if skill is None:
            error = f"Unknown skill: {name}"
            return AgentToolResult(
                content=[
                    TextContent(text=f"{error}. {build_skill_index(skills)}"),
                ],
            )
        return AgentToolResult(
            content=[
                TextContent(text=format_skill(skill)),
            ]
        )

    return AgentTool(
        name="use_skill",
        label="use_skill",
        description=(
            "Load a skill: focused step-by-step instructions for a specific "
            f"kind of task. {build_skill_index(skills)}"
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The skill to load",
                    "enum": sorted(skills_by_name),
                },
            },
            "required": ["name"],
        },  # ty:ignore[invalid-argument-type]
        execute_fn=executor,
    )
