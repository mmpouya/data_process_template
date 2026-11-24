import asyncio

from ddgs import DDGS
from tau_agent.messages import TextContent
from tau_agent.tools import AgentTool, AgentToolResult


async def web_search_executor(tool_call_id, arguments, signal=None, on_update=None):
    """Executor matching the ToolExecutor protocol: takes JSON-like args, returns AgentToolResult."""
    del tool_call_id, on_update

    query = arguments.get("query")
    max_results = arguments.get("max_results", 5)

    if not isinstance(query, str) or not query.strip():
        return AgentToolResult(
            content=[
                TextContent(text="Error: 'query' argument must be a non-empty string.")
            ],
        )

    def _run_search() -> list[dict]:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=int(max_results)))

    try:
        results = await asyncio.to_thread(_run_search)
    except Exception as exc:
        return AgentToolResult(
            content=[TextContent(text=f"Web search failed: {exc}")],
        )

    if not results:
        return AgentToolResult(
            content=[TextContent(text=f"No results found for query: {query!r}")],
        )

    formatted = "\n\n".join(
        f"[{r['title']}]({r['href']})\n{r['body']}" for r in results
    )
    return AgentToolResult(
        content=[TextContent(text=f"## Search Results\n\n{formatted}")],
    )


web_search_tool = AgentTool(
    name="web_search",
    label="web_search",
    description=(
        "Search the web for up-to-date information (news, docs, facts) "
        "and return the top matching results with title, URL, and snippet."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to run, e.g. 'latest LLM benchmarks 2026'.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return.",
                "default": 5,
            },
        },
        "required": ["query"],
    },
    execute_fn=web_search_executor,
    prompt_snippet="Use `web_search` when you need current facts beyond your training data.",
    prompt_guidelines=(
        "Prefer specific, keyword-focused queries over full sentences.",
        "Do not call this tool for simple arithmetic or well-known static facts.",
    ),
)
