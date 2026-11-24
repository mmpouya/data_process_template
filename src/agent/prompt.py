from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from .exceptions import AgentPromptError
from .skill import Skill, build_skill_index

_PLACEHOLDER = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
_PACKAGE_DIR = Path(__file__).parent
DEFAULT_PROMPT_DIR = _PACKAGE_DIR / "prompts"


class PromptLibrary:
    """Nested prompt texts loaded from a YAML file, addressed by dotted keys."""

    def __init__(
        self,
        prompts: Mapping[str, object],
        agent_type: str,
        *,
        source: Path | None = None,
    ) -> None:
        self.prompts = dict(prompts)
        self.source = source
        self.agent_type = agent_type
        self.tools_names = list(
            self.prompts.get(self.agent_type).get("tools").keys()  # ty:ignore[unresolved-attribute]
        )
        self.guideline_names = list(
            self.prompts.get(self.agent_type).get("guidelines").keys()  # ty:ignore[unresolved-attribute]
        )

    def build_system_prompt(
        self, skills: Sequence[Skill] | None = None, **extra_variables: str
    ) -> str:
        """Assemble the Researcher system prompt from the prompt library.

        The tool list is intentionally static (backend search, read_item and the
        memory pair are always described) so the model knows the full surface
        even when a run omits some tools, e.g. memory without a user id.
        """

        tool_descriptions = [
            (tool, self.get(f"{self.agent_type}.tools.{tool}"))
            for tool in self.tools_names
        ]
        if skills and "use_skill" not in self.tools_names:
            tool_descriptions.append(
                ("use_skill", self.get(f"{self.agent_type}.tools.use_skill"))
            )
        tool_lines = "\n".join(
            self.get(f"{self.agent_type}.tool_line", name=name, description=description)
            for name, description in tool_descriptions
        )

        guidelines = [
            self.get(f"{self.agent_type}.guidelines.{guide_name}").rstrip()
            for guide_name in self.guideline_names
        ]

        system = self.get(
            f"{self.agent_type}.system",
            tool_lines=tool_lines,
            guideline_lines="\n".join(guidelines),
            **extra_variables,
        ).rstrip()

        if skills:
            section = self.get(
                f"{self.agent_type}.skills_section",
                skill_index=build_skill_index(skills),
            ).rstrip()
            system = f"{system}\n\n{section}"
        return system

    @classmethod
    def from_yaml(cls, agent_type: str) -> PromptLibrary:
        """Load a prompt library from a YAML file."""
        root = Path(DEFAULT_PROMPT_DIR)
        if not root.is_dir():
            raise AgentPromptError(f"Prompt directory not found: {root}")
        prompt_file = root / f"{agent_type}.yaml"

        if not prompt_file.is_file():
            raise AgentPromptError(f"Prompt file not found: {prompt_file}")
        data = yaml.safe_load(prompt_file.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise AgentPromptError(f"Prompt file must be a YAML mapping: {prompt_file}")
        return cls(data, agent_type=agent_type, source=prompt_file)

    def get(self, key: str, **variables: str) -> str:
        """Return the prompt at a dotted key with `{placeholders}` filled in.

        Raises ``AgentPromptError`` when the key does not exist, points at a
        non-string node, or the prompt references a variable not provided.
        Extra variables are ignored so callers can pass one shared context.
        """
        node = self.prompts
        for part in key.split("."):
            if not isinstance(node, Mapping) or part not in node:
                raise AgentPromptError(f"Unknown prompt key: {key}{self._origin()}")
            node = node[part]
        if not isinstance(node, str):
            raise AgentPromptError(
                f"Prompt key is not a string: {key} — use a full dotted path"
                f"{self._origin()}"
            )

        def substitute(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in variables:
                raise AgentPromptError(
                    f"Prompt {key!r} needs variable {name!r}{self._origin()}"
                )
            return str(variables[name])

        return _PLACEHOLDER.sub(substitute, node)

    def _origin(self) -> str:
        return f" (loaded from {self.source})" if self.source else ""


# if __name__ == "__main__":
#     test_prompt = PromptLibrary.from_yaml("concept_extractor")
#     print(list(test_prompt.prompts.get("concept_extractor").get("tools").keys()))
