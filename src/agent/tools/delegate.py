"""Sub-agent delegation tool (`delegate`).

The orchestrator calls `delegate` to spawn a sub-agent for a multi-step
task. The sub-agent is a nested tau ``AgentHarness`` running the exact same
system prompt and tool set as the orchestrator — minus `delegate` itself,
so delegation cannot recurse. The sub-agent's events (thinking, answering,
tool calls, tool results) are forwarded to the same SSE stream, tagged with
metadata ``{"agent": "sub"}`` so clients can tell orchestrator output from
sub-agent output. The sub-agent's final answer becomes the tool result.

The actual sub-harness is provided by the caller through the `runner`
closure (built by the harness with its provider/model/tools); the tool
itself only validates arguments and delegates execution.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from tau_agent.messages import TextContent
from tau_agent.tools import AgentTool, AgentToolResult, ToolCancellationToken

DelegateRunner = Callable[
    [str, str | None, ToolCancellationToken | None], Awaitable[AgentToolResult]
]


async def _not_available_executor(
    tool_call_id, arguments, signal=None, on_update=None
) -> AgentToolResult:
    del tool_call_id, arguments, signal, on_update
    return AgentToolResult(
        content=[
            TextContent(
                text="delegate is not available in this context (no sub-agent runner)."
            )
        ]
    )


def create_delegate_tool(runner: DelegateRunner | None = None) -> AgentTool:
    """Create the `delegate` tool that spawns a sub-agent.

    ``runner`` is an optional async callable ``(task, instructions, signal)``
    returning the sub-agent's result; without it the tool reports that it is
    unavailable (used by the catalog listing endpoint).
    """

    async def execute_fn(tool_call_id, arguments, signal=None, on_update=None):
        if runner is None:
            return await _not_available_executor(
                tool_call_id, arguments, signal, on_update
            )
        del tool_call_id, on_update
        args = arguments or {}
        task = str(args.get("task", "")).strip()
        instructions = str(args.get("instructions", "")).strip() or None
        if not task:
            return AgentToolResult(
                content=[
                    TextContent(text="delegate requires a non-empty 'task' argument.")
                ]
            )
        return await runner(task, instructions, signal)

    return AgentTool(
        name="delegate",
        label="Delegate to Sub-Agent",
        description=(
            "Spawn a sub-agent to handle a multi-step task and return its "
            "final answer. The sub-agent has the same skills and tools as "
            "you, except this delegation tool. Use it when the user's "
            "request involves several independent steps (e.g. identify an "
            "author AND find their books AND check library availability) — "
            "give the sub-agent ONE clearly bounded task at a time, written "
            "in Persian, with enough context and the exact output you need. "
            "Synthesize the sub-agent's result into your final answer."
        ),
        parameters={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "The sub-task for the sub-agent, in Persian, with all "
                        "needed context and the expected outcome."
                    ),
                },
                "instructions": {
                    "type": "string",
                    "description": (
                        "Optional extra constraints for the sub-agent, e.g. "
                        "which sources to prefer or the answer format."
                    ),
                },
            },
            "required": ["task"],
        },
        execute_fn=execute_fn,
        prompt_snippet=(
            "Before delegating, announce it to the user (answering state), "
            "then call delegate with one clearly bounded Persian task."
        ),
    )
