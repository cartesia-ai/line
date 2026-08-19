"""Web Research Agent with Keenable and Cartesia Line SDK.

Keenable is keyless by default: with no API key the agent calls the public
endpoints (rate-limited). Set KEENABLE_API_KEY to switch to the authenticated
endpoints and lift the rate limit.
"""

from datetime import datetime
import os
from typing import Annotated, Optional

import httpx
from loguru import logger

from line.llm_agent import LlmAgent, LlmConfig, ToolEnv, end_call, loopback_tool
from line.voice_agent_app import AgentEnv, CallRequest, VoiceAgentApp

KEENABLE_BASE_URL = "https://api.keenable.ai"

SYSTEM_PROMPT_TEMPLATE = """Today is {today}. You are a sharp, fast research assistant on a live voice call.

You have two web tools powered by Keenable:

1. web_search — Find relevant pages across the web. Use for questions about current events, \
facts, prices, people, or anything that needs fresh data. Start here for most questions.

2. web_fetch — Pull the full content from a specific URL as clean text. Use when a search \
snippet is too thin to answer confidently, or when the user mentions a specific link.

Your workflow: search first, scan the snippets. If you can answer from snippets alone, do it \
immediately. If a result looks right but you need more detail, fetch that page and then answer. \
Don't fetch unless you need to.

When answering:
- Lead with the answer, not the preamble. No "Great question" or "Let me look that up."
- Keep it to two or three sentences unless the user asks you to go deeper.
- Name your source naturally when it matters. "According to Reuters" beats rattling off URLs.
- If results conflict or seem stale, say so. Don't fake confidence.
- If you genuinely can't find it, say that and suggest how the user could refine.

Use end_call when the user wraps up.

CRITICAL: This is a voice call. Speak in plain, natural sentences only. No markdown, no bullet \
points, no numbered lists, no asterisks, no dashes, no special characters of any kind."""

INTRODUCTION = (
    "Hey! I'm your research assistant, powered by Keenable and Cartesia. "
    "Ask me anything and I'll dig it up live. What do you want to know?"
)

MAX_OUTPUT_TOKENS = 600
TEMPERATURE = 0.7
MAX_RESULTS = 5
FETCH_MAX_CHARS = 3000
# Keenable returns whole-page text on every search result; a voice turn only
# needs enough to decide whether to fetch the page.
SNIPPET_MAX_CHARS = 500


class KeenableTools:
    """Holds a single httpx.AsyncClient so the connection pool is reused across
    tool calls. Keyless by default; an optional API key lifts the rate limit."""

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = (api_key or "").strip()
        self._client = httpx.AsyncClient(timeout=15.0)

    def _headers(self) -> dict:
        headers = {
            "Accept": "application/json",
            "User-Agent": "keenable-cartesia-line",
            "X-Keenable-Title": "Cartesia Line",
        }
        if self._api_key:
            headers["X-API-Key"] = self._api_key
        return headers

    @loopback_tool
    async def web_search(
        self,
        ctx: ToolEnv,
        query: Annotated[
            str,
            "The search query. Be specific and include key terms.",
        ],
        published_after: Annotated[
            Optional[str],
            "Optional date filter (YYYY-MM-DD). Only return pages published on or after this date.",
        ] = None,
    ) -> str:
        """Search the web for current information.
        Use when you need up-to-date facts, news, or any information that requires factual accuracy."""
        logger.info(f"Performing Keenable web search: '{query}'")

        path = "/v1/search" if self._api_key else "/v1/search/public"
        payload: dict = {"query": query, "mode": "pro"}
        if published_after is not None:
            payload["published_after"] = published_after

        try:
            resp = await self._client.post(
                f"{KEENABLE_BASE_URL}{path}", json=payload, headers=self._headers()
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])[:MAX_RESULTS]
            if not results:
                return "No relevant information found."

            parts = [f"Search Results for: '{query}'\n"]
            for i, result in enumerate(results):
                parts.append(f"\n--- Source {i + 1}: {result.get('title', 'Untitled')} ---\n")
                # Keenable returns both `snippet` and `description`: `snippet` carries
                # the page text and `description` is the page's meta description,
                # which is empty for most pages.
                page_text = str(result.get("snippet") or result.get("description") or "")
                snippet = " ".join(page_text.split())[:SNIPPET_MAX_CHARS]
                if snippet:
                    parts.append(f"{snippet}\n")
                parts.append(f"URL: {result.get('url', '')}\n")

            logger.info(f"Search completed: {len(results)} sources found")
            return "".join(parts)

        except Exception as e:
            logger.error(f"Keenable search failed: {e}")
            return f"Web search failed: {e}"

    @loopback_tool
    async def web_fetch(
        self,
        ctx: ToolEnv,
        url: Annotated[
            str,
            "The URL to fetch content from.",
        ],
    ) -> str:
        """Fetch the full content of a webpage given its URL, as clean text.
        Use when you need detailed information from a specific page found via web_search."""
        logger.info(f"Fetching content from: '{url}'")

        path = "/v1/fetch" if self._api_key else "/v1/fetch/public"
        try:
            resp = await self._client.get(
                f"{KEENABLE_BASE_URL}{path}", params={"url": url}, headers=self._headers()
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("content", "")
            if not content:
                return "The page was reached but no readable content was found."

            if len(content) > FETCH_MAX_CHARS:
                content = content[:FETCH_MAX_CHARS] + "\n\n[Content truncated]"

            logger.info(f"Fetch completed: {len(content)} characters from {url}")
            title = data.get("title", "")
            header = f"{title}\n\n" if title else ""
            return f"Content from {url}:\n\n{header}{content}"

        except Exception as e:
            logger.error(f"Keenable fetch failed: {e}")
            return f"Content fetch failed: {e}"


async def get_agent(env: AgentEnv, call_request: CallRequest):
    today = datetime.now().strftime("%Y-%m-%d")
    # Keenable is keyless by default; KEENABLE_API_KEY is optional (lifts the rate limit).
    keenable = KeenableTools(api_key=os.environ.get("KEENABLE_API_KEY"))
    return LlmAgent(
        model="openai/gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        tools=[keenable.web_search, keenable.web_fetch, end_call],
        config=LlmConfig(
            system_prompt=SYSTEM_PROMPT_TEMPLATE.format(today=today),
            introduction=INTRODUCTION,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=TEMPERATURE,
        ),
    )


app = VoiceAgentApp(get_agent=get_agent)

if __name__ == "__main__":
    app.run()
