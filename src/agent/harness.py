import logging
from collections.abc import Callable, Coroutine
from typing import Any

from tau_agent import AgentTool
from tau_agent.events import MessageEndEvent, ToolExecutionEndEvent
from tau_agent.harness import AgentHarness, AgentHarnessConfig
from tau_agent.messages import AssistantMessage, TextContent, UserMessage
from tau_agent.tools import AgentToolResult
from tau_ai.env import OpenAICompatibleConfig
from tau_ai.openai_compatible import OpenAICompatibleProvider

from ..core.config import get_settings
from .exceptions import AgentHarnessError
from .prompt import PromptLibrary
from .publisher import EventPublisher
from .schema import LLMSendMessageTypeEnum, ResultMessage, UserPromptSchema
from .session import AgentSession
from .skill import DEFAULT_SKILLS_DIR, load_skills
from .tools.ask_user import create_ask_user_tool
from .tools.delegate import create_delegate_tool
from .tools.fetch_url import fetch_url_tool
from .tools.timeout import apply_tool_timeouts
from .tools.use_skill import create_use_skill_tool
from .tools.web_search import web_search_tool

logger = logging.getLogger(__name__)


def build_base_tools() -> list[AgentTool]:
    return [web_search_tool, fetch_url_tool]


all_tools = build_base_tools()
TOOLS_BY_NAME: dict[str, AgentTool] = {tool.name: tool for tool in all_tools}


class Harness:
    def __init__(
        self,
        agent_name: str,
        queue_handler: Callable[[ResultMessage], Coroutine[Any, Any, None]],
        session: AgentSession | None = None,
        tools: list[AgentTool] = [],
    ):
        self.agent_name = agent_name
        self.queue_handler = queue_handler
        self.session = session
        self.tools = tools
        self._provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key=get_settings().OPENAI_API_KEY,
                base_url=get_settings().OPENAI_BASE_URL,
                reasoning_effort="low",
            )
        )
        self._prompt: PromptLibrary = self._system_prompt()
        self._skills = load_skills(DEFAULT_SKILLS_DIR)
        self._skills_for_prompt = (
            self._skills if "use_skill" in self._prompt.tools_names else None
        )
        self.tools = self._build_tools()

        logger.info(
            f"[HARNESS AGENT] agent {self.agent_name} initialized with these tools: {[t.name for t in self.tools]}"
        )

    def _system_prompt(self) -> PromptLibrary:
        self._prompt = PromptLibrary.from_yaml(self.agent_name)
        return self._prompt

    def _build_tools(self) -> list[AgentTool]:
        """Resolve the YAML-declared tools + closure-bound orchestration tools."""
        tools: list[AgentTool] = []
        for name in self._prompt.tools_names:
            tool = TOOLS_BY_NAME.get(name)
            if tool is None:
                tool = TOOLS_BY_NAME.get(f"{name}_tool")
            if tool is None:
                tool = TOOLS_BY_NAME.get(name.removesuffix("_tool"))
            if tool is not None:
                tools.append(tool)
        if not tools and self._prompt.tools_names:
            tools = list(all_tools)
        if self.session is not None and "ask_user" in self._prompt.tools_names:
            tools.append(create_ask_user_tool(self.session))
        if self._skills_for_prompt:
            tools.append(create_use_skill_tool(self._skills_for_prompt))
        if "delegate" in self._prompt.tools_names:
            tools.append(create_delegate_tool(runner=self._delegate_runner))
        settings = get_settings()
        return apply_tool_timeouts(
            tools,
            default_seconds=settings.TOOL_TIMEOUT,
            overrides={"delegate": settings.DELEGATE_TIMEOUT},
        )

    async def _delegate_runner(
        self, task: str, instructions: str | None, signal=None
    ) -> AgentToolResult:
        """Run a sub-agent with the same prompt and tools minus `delegate`."""
        full_task = task if not instructions else f"{task}\n\n{instructions}"
        sub_tools = [tool for tool in self.tools if tool.name != "delegate"]
        sub_harness = AgentHarness(
            AgentHarnessConfig(
                self._provider,
                model=get_settings().OPENAI_MODEL_NAME,
                system=self._prompt.build_system_prompt(skills=self._skills_for_prompt),
                tools=sub_tools,
            )
        )
        publisher = EventPublisher(self.queue_handler, metadata_extra={"agent": "sub"})
        final_text = ""
        sub_stream = sub_harness.prompt(full_task)
        async for event in sub_stream:
            if signal is not None and signal.is_cancelled():
                sub_harness.cancel()
                break
            await publisher(event)
            if isinstance(event, MessageEndEvent) and isinstance(
                event.message, AssistantMessage
            ):
                final_text = "".join(
                    block.text
                    for block in event.message.content
                    if isinstance(block, TextContent)
                )

        if not final_text.strip():
            final_text = "(sub-agent produced no text answer)"
        return AgentToolResult(content=[TextContent(text=final_text)])

    async def run(self, messages: list[UserPromptSchema]) -> None:
        user_query = str(messages[-1].content).strip()
        self.harness = self._make_harness()
        self._replay_history(self.harness, messages)
        publisher = EventPublisher(self.queue_handler)
        async for event in self.harness.prompt(user_query):
            await publisher(event)

    async def run_collect(self, messages: list[UserPromptSchema]) -> str:
        """Run the agent and return the final assistant text.

        Unlike :meth:`run`, nothing is streamed to the queue handler; the
        method is meant for internal batch tasks (``AITaskManager``) that need
        a direct result value.
        """
        user_query = str(messages[-1].content).strip()
        harness = self._make_harness()
        self._replay_history(harness, messages)
        final_text = ""
        async for event in harness.prompt(user_query):
            if isinstance(event, ToolExecutionEndEvent) and event.is_error:
                result = event.result
                if isinstance(result.content[-1], TextContent):
                    raise AgentHarnessError(result.content[-1].text)
                raise AgentHarnessError(str(event))
            if isinstance(event, MessageEndEvent) and isinstance(
                event.message, AssistantMessage
            ):
                final_text = "".join(
                    block.text
                    for block in event.message.content
                    if isinstance(block, TextContent)
                )
        if not final_text.strip():
            raise AgentHarnessError("Agent produced no text answer")
        return final_text.strip()

    def _make_harness(self) -> AgentHarness:
        return AgentHarness(
            AgentHarnessConfig(
                self._provider,
                model=get_settings().OPENAI_MODEL_NAME,
                system=self._prompt.build_system_prompt(skills=self._skills_for_prompt),
                tools=self.tools,
            )
        )

    def _replay_history(
        self, harness: AgentHarness, messages: list[UserPromptSchema]
    ) -> None:
        for msg in messages[:-1]:
            content = str(msg.content).strip()
            if not content:
                continue
            if msg.role == LLMSendMessageTypeEnum.USER:
                harness.append_message(UserMessage(content=content))
            elif msg.role == LLMSendMessageTypeEnum.ASSISTANT:
                harness.append_message(
                    AssistantMessage(content=[TextContent(text=content)])
                )
