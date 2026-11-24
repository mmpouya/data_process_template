"""Human-in-the-loop question tool.

The agent calls `ask_user` at critical decision points (assessments,
evaluations, interpretation choices). The tool publishes the question to the
SSE stream, then blocks until the user answers through the `/chat/answer`
endpoint. The user's answer becomes the tool result and the agent continues.
"""

from __future__ import annotations

import asyncio
import json
import uuid

from tau_agent import AgentToolResult
from tau_agent.messages import TextContent
from tau_agent.tools import AgentTool

from ...core.config import get_settings
from ..schema import ResultMessage
from ..session import AgentSession

QUESTION_TYPES = ("single_choice", "multi_choice", "explanation", "yes_no")
_CHOICE_TYPES = ("single_choice", "multi_choice", "yes_no")


def _format_answer(answer) -> str:
    if isinstance(answer, str):
        return answer
    if isinstance(answer, (list, tuple)):
        return "، ".join(str(item) for item in answer)
    return json.dumps(answer, ensure_ascii=False)


async def _ask_user_execute(
    session: AgentSession,
    tool_call_id: str,
    arguments: dict,
    signal=None,
    on_update=None,
) -> AgentToolResult:
    question: str = str(arguments.get("question", "")).strip()
    qtype: str = str(arguments.get("type", "explanation")).strip()
    raw_options = arguments.get("options")
    if isinstance(raw_options, str):
        try:
            parsed = json.loads(raw_options)
        except json.JSONDecodeError:
            parsed = None
        options = parsed if isinstance(parsed, list) else [raw_options]
    elif isinstance(raw_options, list):
        options = raw_options
    else:
        options = []
    options = [str(o).strip() for o in options if str(o).strip()]
    context: str = str(arguments.get("context", "")).strip()

    if not question:
        return AgentToolResult(
            content=[TextContent(text="no question in input arguments")]
        )
    if qtype not in QUESTION_TYPES:
        return AgentToolResult(
            content=[
                TextContent(
                    text=f"invalid question type {qtype!r}; "
                    f"must be one of {', '.join(QUESTION_TYPES)}"
                )
            ]
        )
    if qtype in _CHOICE_TYPES and not options:
        return AgentToolResult(
            content=[
                TextContent(
                    text=f"question type {qtype!r} requires a non-empty options list"
                )
            ]
        )

    question_id = uuid.uuid4().hex
    session.pending_question_id = question_id
    await session.publish(
        ResultMessage(
            state="question",
            chunk=question,
            metadata={
                "session_id": session.session_id,
                "question_id": question_id,
                "type": qtype,
                "options": options,
                "context": context or None,
            },
        )
    )

    try:
        payload = await asyncio.wait_for(
            session.answer_queue.get(), timeout=get_settings().QUESTION_TIMEOUT
        )
    except asyncio.TimeoutError:
        return AgentToolResult(
            content=[
                TextContent(
                    text=(
                        "the user did not answer within the time limit; "
                        "decide on your own or ask again later"
                    )
                )
            ]
        )

    answer = payload.get("answer") if isinstance(payload, dict) else payload
    return AgentToolResult(content=[TextContent(text=_format_answer(answer))])


def create_ask_user_tool(session: AgentSession | None = None) -> AgentTool:
    """Create the `ask_user` tool bound to a live agent session.

    ``session`` is optional: without a live session the tool reports that it
    is unavailable (used by the catalog listing endpoint).
    """

    async def execute_fn(tool_call_id, arguments, signal=None, on_update=None):
        if session is None:
            return AgentToolResult(
                content=[
                    TextContent(text="ask_user is unavailable without a live session.")
                ]
            )
        return await _ask_user_execute(
            session, tool_call_id, arguments, signal, on_update
        )

    return AgentTool(
        name="ask_user",
        label="Ask User",
        description=(
            "Asks the user a critical question during the research and waits for "
            "their answer before continuing. Use it for assessments, evaluations, "
            "and choices about the research path (e.g. which interpretation to "
            "follow, whether to include a weak hadith, which topic to prioritize)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user, in Persian.",
                },
                "type": {
                    "type": "string",
                    "enum": list(QUESTION_TYPES),
                    "description": (
                        "single_choice: pick exactly one option; "
                        "multi_choice: pick any number of options; "
                        "yes_no: confirm or reject; "
                        "explanation: free-text answer."
                    ),
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Answer choices; required for single_choice, multi_choice "
                        "and yes_no, ignored for explanation."
                    ),
                },
                "context": {
                    "type": "string",
                    "description": "Optional short explanation of why the question is asked.",
                },
            },
            "required": ["question", "type"],
        },
        execute_fn=execute_fn,
        prompt_snippet=(
            "Use `ask_user` whenever a critical decision point is reached and the "
            "user's judgment is needed before continuing the research."
        ),
    )
