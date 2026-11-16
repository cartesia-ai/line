"""Web Research Agent with SpecStream and Cartesia Line SDK."""

from datetime import datetime
import os
from typing import Annotated, Literal, Optional

import aiohttp
from loguru import logger

from line.llm_agent import LlmAgent, LlmConfig, ToolEnv, end_call, loopback_tool
from line.voice_agent_app import AgentEnv, CallRequest, VoiceAgentApp

SYSTEM_PROMPT_TEMPLATE = """Today is {today}. You are a fast, accurate research assistant on a voice call.

You have two web tools powered by SpecStream:

1. web_search: Search the live web for current events, facts, prices, or any topic that needs fresh data. \
Start here for most questions.

2. web_extract: Extract the main text of a specific URL as markdown. Use when a search snippet is too thin \
to answer confidently, or when the user asks about a specific link.

Your workflow: search first, scan the answer and snippets. If you can answer from the search response alone, \
do it immediately. If a result looks right but you need more detail, extract that page and then answer. \
Do not extract unless you need to.

When answering:
- Lead with the answer, not preamble. No "Great question" or "Let me look that up."
- Keep it to two or three sentences unless the user asks for more detail.
- Name your source naturally when it matters. "According to Wikipedia" works better than reading URLs aloud.
- If results conflict or seem stale, say so.
- If you cannot find the answer, say so and suggest how to refine the question.

Use end_call when the user wraps up.

CRITICAL: This is a voice call. Speak in plain, natural sentences only. No markdown, no bullet points, \
no numbered lists, no asterisks, no dashes, no special characters of any kind."""

INTRODUCTION = (
    "Hey! I'm your research assistant, powered by SpecStream and Cartesia. "
    "Ask me anything and I'll look it up live. What would you like to know?"
)

DEFAULT_BASE_URL = "https://api.specstream.ai"
MAX_OUTPUT_TOKENS = 600
TEMPERATURE = 0.7
SEARCH_NUM_RESULTS = 5
EXTRACT_MAX_CHARS = 3000
REQUEST_TIMEOUT_SECONDS = 30


class SpecStreamTools:
    """Holds a shared aiohttp session for SpecStream Search and Extract calls."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        """Initialize SpecStream tools.

        Args:
            api_key: SpecStream API key. Falls back to SPECSTREAM_API_KEY env var.
            base_url: SpecStream API base URL. Defaults to https://api.specstream.ai.
        """
        self._api_key = api_key if api_key is not None else os.environ.get("SPECSTREAM_API_KEY")
        raw_base = base_url if base_url is not None else os.environ.get("SPECSTREAM_BASE_URL")
        self._base_url = (raw_base or DEFAULT_BASE_URL).rstrip("/")
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        """Close the underlying HTTP session if open."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    @staticmethod
    async def _read_error(response: aiohttp.ClientResponse) -> str:
        try:
            payload = await response.json(content_type=None)
        except Exception:
            return (await response.text()) or response.reason or "unknown error"
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict) and error.get("message"):
                return str(error["message"])
            if payload.get("detail"):
                return str(payload["detail"])
        return response.reason or "unknown error"

    @loopback_tool
    async def web_search(
        self,
        ctx: ToolEnv,
        query: Annotated[
            str,
            "The search query in plain words. Be specific and include key terms.",
        ],
        speed: Annotated[
            Literal["fast", "default"],
            "Search tier: 'fast' for low-latency snippets or 'default' for full answer synthesis.",
        ] = "fast",
        news: Annotated[
            bool,
            "Set to true to search recent news items instead of general web results.",
        ] = False,
    ) -> str:
        """Search the live web for current facts, news, or information."""
        if not self._api_key:
            logger.warning("SpecStream web_search called without SPECSTREAM_API_KEY")
            return "Web search failed: SPECSTREAM_API_KEY is not set."

        tier = speed if speed in ("fast", "default") else "fast"
        params: dict[str, str] = {
            "q": query,
            "speed": tier,
            "num": str(SEARCH_NUM_RESULTS),
        }
        if news:
            params["news"] = "true"

        logger.info(f"Performing SpecStream web search ({tier}): '{query}'")

        try:
            session = await self._get_session()
            async with session.get(
                f"{self._base_url}/v1/web-search",
                headers=self._auth_headers(),
                params=params,
            ) as response:
                if response.status != 200:
                    detail = await self._read_error(response)
                    logger.error(f"SpecStream search failed ({response.status}): {detail}")
                    return f"Web search failed (HTTP {response.status}): {detail}"
                payload = await response.json()
        except Exception as e:
            logger.error(f"SpecStream search failed: {e}")
            return f"Web search failed: {e}"

        data = payload.get("data", {}) if isinstance(payload, dict) else {}
        answer = data.get("answer")
        sources = data.get("sources") or []

        if not answer and not sources:
            return "No relevant information found."

        content_parts = [f"Search Results for: '{query}'\n"]
        if isinstance(answer, dict) and answer.get("value"):
            answer_page = answer.get("page") or ""
            suffix = f" (source: {answer_page})" if answer_page else ""
            content_parts.append(f"\nAnswer: {answer['value']}{suffix}\n")

        for i, source in enumerate(sources[:SEARCH_NUM_RESULTS]):
            title = source.get("title") or source.get("url") or "Untitled"
            content_parts.append(f"\n--- Source {i + 1}: {title} ---\n")
            snippet = source.get("snippet") or source.get("excerpt")
            if snippet:
                content_parts.append(f"{snippet}\n")
            if source.get("published"):
                content_parts.append(f"Published: {source['published']}\n")
            if source.get("url"):
                content_parts.append(f"URL: {source['url']}\n")

        took_ms = data.get("took_ms", 0)
        logger.info(f"Search completed: {len(sources)} sources found in {took_ms}ms")
        return "".join(content_parts)

    @loopback_tool
    async def web_extract(
        self,
        ctx: ToolEnv,
        url: Annotated[
            str,
            "The http or https URL to extract main text from.",
        ],
    ) -> str:
        """Extract the main content of a webpage as markdown given its URL."""
        if not self._api_key:
            logger.warning("SpecStream web_extract called without SPECSTREAM_API_KEY")
            return "Content extraction failed: SPECSTREAM_API_KEY is not set."

        logger.info(f"Extracting content from: '{url}'")

        try:
            session = await self._get_session()
            async with session.post(
                f"{self._base_url}/v1/extract",
                headers={**self._auth_headers(), "Content-Type": "application/json"},
                json={"urls": [url], "format": "markdown"},
            ) as response:
                if response.status != 200:
                    detail = await self._read_error(response)
                    logger.error(f"SpecStream extract failed ({response.status}): {detail}")
                    return f"Content extraction failed (HTTP {response.status}): {detail}"
                payload = await response.json()
        except Exception as e:
            logger.error(f"SpecStream extract failed: {e}")
            return f"Content extraction failed: {e}"

        data = payload.get("data", {}) if isinstance(payload, dict) else {}
        results = data.get("results") or []
        if not results:
            return "No content could be extracted from that URL."

        extracted = results[0]
        status = extracted.get("status")
        if status != "ok":
            reason = extracted.get("reason") or status or "unknown error"
            return f"Extraction failed for {url}: {reason}"

        text = extracted.get("text") or ""
        if not text:
            return "The page was reached but no readable content was found."

        if len(text) > EXTRACT_MAX_CHARS:
            text = text[:EXTRACT_MAX_CHARS] + "\n\n[Content truncated]"

        title = extracted.get("title")
        header = f"Extracted content from {title} ({url}):" if title else f"Extracted content from {url}:"
        logger.info(f"Extraction completed: {len(text)} characters from {url}")
        return f"{header}\n\n{text}"


async def get_agent(env: AgentEnv, call_request: CallRequest) -> LlmAgent:
    """Create a voice research agent wired with SpecStream tools.

    Args:
        env: Agent environment provided by the Line runtime.
        call_request: Incoming call metadata and configuration.

    Returns:
        Configured LlmAgent instance.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    specstream = SpecStreamTools(api_key=os.environ.get("SPECSTREAM_API_KEY"))
    return LlmAgent(
        model="openai/gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        tools=[specstream.web_search, specstream.web_extract, end_call],
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
