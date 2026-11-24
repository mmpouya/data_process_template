"""Central per-tool execution timeout wrapper.

Some tools call external HTTP APIs and scraped web sources that can go
silent (no response, hung connection, dead server). Individual tools already
set their own ``httpx`` timeouts and ``tenacity`` retries, but that is not a
hard guarantee: retries multiply the effective duration, ``asyncio.to_thread``
calls (e.g. DuckDuckGo search) are unbounded, and future tools might forget.
``apply_tool_timeouts`` wraps every tool's ``execute_fn`` in
``asyncio.wait_for`` so the run can never stall on a tool.

A timeout returns a **normal** ``AgentToolResult`` (not an error): the tau
loop treats tool exceptions as fatal for the run, while a recoverable text
result lets the model continue the session without that source.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections.abc import Mapping, Sequence

from tau_agent.messages import TextContent
from tau_agent.tools import AgentTool, AgentToolResult

logger = logging.getLogger(__name__)

#: Tools that manage their own long waits and must not be wrapped.
DEFAULT_EXEMPT_TOOLS: frozenset[str] = frozenset({"ask_user"})


def _tool_timeout_result(tool: AgentTool, seconds: float) -> AgentToolResult:
    return AgentToolResult(
        content=[
            TextContent(
                text=(
                    f"Tool {tool.name} timed out after {seconds:.0f} seconds: "
                    "the external source did not respond. Continue without this "
                    "result, or try another source if needed."
                )
            )
        ]
    )


def wrap_tool_timeout(tool: AgentTool, seconds: float) -> AgentTool:
    """Return a copy of ``tool`` whose executor times out after ``seconds``.

    ``seconds`` <= 0 disables the wrapper (returns the tool unchanged).
    """
    if seconds <= 0:
        return tool
    original = tool.execute_fn

    async def execute_fn(tool_call_id, arguments, signal=None, on_update=None):
        try:
            return await asyncio.wait_for(
                original(tool_call_id, arguments, signal, on_update),
                timeout=seconds,
            )
        except TimeoutError:
            logger.warning("tool %s timed out after %.0fs", tool.name, seconds)
            return _tool_timeout_result(tool, seconds)

    return dataclasses.replace(tool, execute_fn=execute_fn)


def apply_tool_timeouts(
    tools: Sequence[AgentTool],
    default_seconds: float,
    overrides: Mapping[str, float] | None = None,
    exempt: set[str] | frozenset[str] = DEFAULT_EXEMPT_TOOLS,
) -> list[AgentTool]:
    """Wrap every non-exempt tool with a hard execution timeout.

    ``overrides`` maps tool names to a specific timeout (e.g. ``delegate``
    gets a longer budget than scraper tools). ``exempt`` tools (such as
    ``ask_user``, which has its own ``QUESTION_TIMEOUT``) are left unwrapped.
    """
    overrides = overrides or {}
    wrapped: list[AgentTool] = []
    for tool in tools:
        if tool.name in exempt:
            wrapped.append(tool)
            continue
        seconds = float(overrides.get(tool.name, default_seconds))
        wrapped.append(wrap_tool_timeout(tool, seconds))
    return wrapped
