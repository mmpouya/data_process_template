import time
from collections.abc import Callable, Coroutine
from typing import Any

from tau_agent.events import (
    AgentEndEvent,
    MessageUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)
from tau_agent.messages import TextContent
from tau_agent.provider_events import (
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)

from .schema import ResultMessage


class EventPublisher:
    """Map tau AgentEvents to ResultMessages on the SSE queue.

    Reused for both the orchestrator run and sub-agent (delegate) runs; the
    optional ``metadata_extra`` marks sub-agent output (``agent: "sub"``)
    while the orchestrator's events keep the original metadata contract.
    """

    def __init__(
        self,
        queue_handler: Callable[[ResultMessage], Coroutine[Any, Any, None]],
        metadata_extra: dict[str, Any] | None = None,
    ) -> None:
        self.queue_handler = queue_handler
        self.metadata_extra = metadata_extra or {}
        self._thinking_start: float | None = None
        self._tool_start: float | None = None
        self._answer_start: float | None = None

    def _meta(self, **fields: Any) -> dict[str, Any]:
        return {**fields, **self.metadata_extra}

    async def __call__(self, event: Any) -> None:
        if isinstance(event, ToolExecutionStartEvent):
            await self.queue_handler(
                ResultMessage(
                    state="Tool Calling",
                    chunk=f"Tool Name: {event.tool_name}",
                    metadata=self._meta(
                        tool=event.tool_name,
                        phase="start",
                        input=dict(event.args),
                    ),
                )
            )
        elif isinstance(event, ToolExecutionEndEvent):
            result = event.result
            if not event.is_error:
                await self.queue_handler(
                    ResultMessage(
                        state=f"Tool {event.tool_name} result",
                        chunk=f"{result.content[-1].model_dump()}",
                        metadata=self._meta(
                            tool=event.tool_name,
                            phase="end",
                            ok=not event.is_error,
                            output=result.content,
                        ),
                    )
                )
            if event.is_error:
                if isinstance(result.content[-1], TextContent):
                    await self.queue_handler(
                        ResultMessage(error=result.content[-1].text)
                    )
                    raise ValueError("[TAU-AGENT - ERROR:] ", result.content[-1].text)
                else:
                    raise ValueError("[TAU-AGENT - ERROR:] ", event.model_dump())
        elif isinstance(event, MessageUpdateEvent):
            await self._handle_message_update(event)
        elif isinstance(event, AgentEndEvent) and not self.metadata_extra:
            # A sub-agent's terminal event must not end the parent stream:
            # only the orchestrator publishes the terminal `done` message.

            await self.queue_handler(ResultMessage(state="done", chunk="", done=True))

    async def _handle_message_update(self, event: MessageUpdateEvent) -> None:
        assistant_event = event.assistant_message_event
        if isinstance(assistant_event, ThinkingDeltaEvent):
            await self.queue_handler(
                ResultMessage(
                    state="thinking",
                    chunk=assistant_event.delta,
                    metadata=self._meta(),
                )
            )
        elif isinstance(assistant_event, TextDeltaEvent):
            await self.queue_handler(
                ResultMessage(
                    state="answering",
                    chunk=assistant_event.delta,
                    metadata=self._meta(),
                )
            )
        elif isinstance(assistant_event, ThinkingStartEvent):
            self._thinking_start = time.perf_counter()
        elif isinstance(assistant_event, ThinkingEndEvent):
            if self._thinking_start is not None:
                await self.queue_handler(
                    ResultMessage(
                        state="thinking_time",
                        chunk=(
                            f"Elapsed time: "
                            f"{time.perf_counter() - self._thinking_start:.4f} seconds"
                        ),
                        metadata=self._meta(),
                    )
                )
        elif isinstance(assistant_event, ToolCallStartEvent):
            self._tool_start = time.perf_counter()
        elif isinstance(assistant_event, ToolCallEndEvent):
            if self._tool_start is not None:
                await self.queue_handler(
                    ResultMessage(
                        state="tool_time",
                        chunk=(
                            f"Elapsed time: "
                            f"{time.perf_counter() - self._tool_start:.4f} seconds"
                        ),
                        metadata=self._meta(),
                    )
                )
        elif isinstance(assistant_event, TextStartEvent):
            self._answer_start = time.perf_counter()
        elif (
            isinstance(assistant_event, TextEndEvent) and self._answer_start is not None
        ):
            await self.queue_handler(
                ResultMessage(
                    state="answering_time",
                    chunk=(
                        f"Elapsed time: "
                        f"{time.perf_counter() - self._answer_start:.4f} seconds"
                    ),
                    metadata=self._meta(),
                )
            )
