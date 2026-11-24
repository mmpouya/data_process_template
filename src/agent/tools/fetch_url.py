import asyncio
import html
import re

import httpx
from tau_agent.messages import TextContent
from tau_agent.tools import AgentTool, AgentToolResult

MAX_CONTENT_CHARS = 8000
TIMEOUT_SECONDS = 30.0

_TAG_RE = re.compile(r"<[^>]+>")
_BLOCK_RE = re.compile(r"(?is)<(script|style|noscript|svg|head|template).*?</\1>")
_WHITESPACE_RE = re.compile(r"\s+")

_USER_AGENT = "parsa-etl-agent/1.0"


def _strip_html(raw: str) -> str:
    text = _BLOCK_RE.sub(" ", raw)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _fetch(url: str) -> httpx.Response:
    with httpx.Client(
        timeout=TIMEOUT_SECONDS,
        follow_redirects=True,
        headers={"User-Agent": _USER_AGENT},
    ) as client:
        return client.get(url)


async def fetch_url_executor(tool_call_id, arguments, signal=None, on_update=None):
    """Fetch a webpage and return its readable text content."""
    del tool_call_id, on_update

    url = arguments.get("url")
    if not isinstance(url, str) or not url.strip():
        return AgentToolResult(
            content=[
                TextContent(text="Error: 'url' argument must be a non-empty string.")
            ],
        )

    try:
        response = await asyncio.to_thread(_fetch, url)
    except Exception as exc:
        return AgentToolResult(
            content=[TextContent(text=f"Failed to fetch {url}: {exc}")],
        )

    if response.status_code >= 400:
        return AgentToolResult(
            content=[
                TextContent(
                    text=f"HTTP {response.status_code} while fetching {url}"
                )
            ],
        )

    text = _strip_html(response.text)
    if not text:
        return AgentToolResult(
            content=[TextContent(text=f"No readable text found at {url}")],
        )
    if len(text) > MAX_CONTENT_CHARS:
        text = text[:MAX_CONTENT_CHARS] + "\n...[truncated]"
    return AgentToolResult(content=[TextContent(text=text)])


fetch_url_tool = AgentTool(
    name="fetch_url",
    label="fetch_url",
    description=(
        "Fetch a webpage URL and return its readable text content. "
        "Use when you need the actual content of an original website page."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The full URL of the webpage to fetch.",
            },
        },
        "required": ["url"],
    },
    execute_fn=fetch_url_executor,
    prompt_snippet="Use `fetch_url` when you need the actual content of a webpage.",
    prompt_guidelines=(
        "Only fetch URLs provided in the task or discovered during the run.",
        "Content is truncated to the first 8000 characters.",
    ),
)