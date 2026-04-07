"""Web Research Agent with Tavily and Cartesia Line SDK."""

from datetime import datetime
import os
from typing import Annotated

from loguru import logger
from tavily import AsyncTavilyClient

from line.llm_agent import LlmAgent, LlmConfig, ToolEnv, end_call, loopback_tool
from line.voice_agent_app import AgentEnv, CallRequest, VoiceAgentApp

today = datetime.now().strftime("%Y-%m-%d")

SYSTEM_PROMPT = f"""Today is {today}. You are a sharp, fast research assistant on a live voice call.

You have two web tools powered by Tavily:

1. web_search — Find relevant pages across the web. Use for questions about current events, \
facts, prices, people, or anything that needs fresh data. Start here for most questions.

2. web_extract — Pull full content from a specific URL. Use when a search snippet is too \
thin to answer confidently, or when the user mentions a specific link they want you to read.

Your workflow: search first, scan the snippets. If you can answer from snippets alone, do it \
immediately. If a result looks right but you need more detail, extract that page and then answer. \
Don't extract unless you need to.

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
    "Hey! I'm your research assistant, powered by Tavily and Cartesia. "
    "Ask me anything and I'll dig it up live. What do you want to know?"
)

MAX_OUTPUT_TOKENS = 600
TEMPERATURE = 0.7


@loopback_tool
async def web_search(
    ctx: ToolEnv,
    query: Annotated[
        str,
        "The search query. Be specific and include key terms.",
    ],
    time_range: Annotated[
        str,
        "The time range to search for. Use 'day', 'week', 'month', or 'year'.",
    ] = "month",
) -> str:
    """Search the web for current information.
    Use when you need up-to-date facts, news, or any information that requires factual accuracy."""
    logger.info(f"Performing Tavily web search: '{query}'")

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "Web search failed: TAVILY_API_KEY not set."

    try:
        client = AsyncTavilyClient(api_key=api_key, client_source="cartesia-line-agent")
        response = await client.search(
            query=query,
            time_range=time_range,
            search_depth="fast",
            max_results=5,
        )

        results = response.get("results", [])
        if not results:
            return "No relevant information found."

        # Format results for LLM
        content_parts = [f"Search Results for: '{query}'\n"]
        for i, result in enumerate(results):
            score = result.get("score", 0)
            content_parts.append(f"\n--- Source {i + 1}: {result['title']} (relevance: {score:.2f}) ---\n")
            if result.get("content"):
                content_parts.append(f"{result['content']}\n")
            content_parts.append(f"URL: {result['url']}\n")

        response_time = response.get("response_time", 0)
        logger.info(f"Search completed: {len(results)} sources found in {response_time:.2f}s")
        return "".join(content_parts)

    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        return f"Web search failed: {e}"


@loopback_tool
async def web_extract(
    ctx: ToolEnv,
    url: Annotated[
        str,
        "The URL to extract content from.",
    ],
) -> str:
    """Extract the full content of a webpage given its URL.
    Use when you need detailed information from a specific page found via web_search."""
    logger.info(f"Extracting content from: '{url}'")

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "Content extraction failed: TAVILY_API_KEY not set."

    try:
        client = AsyncTavilyClient(api_key=api_key, client_source="cartesia-line-agent")
        response = await client.extract(urls=[url])

        results = response.get("results", [])
        if not results:
            failed = response.get("failed_results", [])
            if failed:
                return f"Extraction failed for {url}: {failed[0].get('error', 'unknown error')}"
            return "No content could be extracted from that URL."

        extracted = results[0]
        raw_content = extracted.get("raw_content", "")
        if not raw_content:
            return "The page was reached but no readable content was found."

        max_chars = 3000
        if len(raw_content) > max_chars:
            raw_content = raw_content[:max_chars] + "\n\n[Content truncated]"

        logger.info(f"Extraction completed: {len(raw_content)} characters from {url}")
        return f"Extracted content from {url}:\n\n{raw_content}"

    except Exception as e:
        logger.error(f"Tavily extract failed: {e}")
        return f"Content extraction failed: {e}"


async def get_agent(env: AgentEnv, call_request: CallRequest):
    return LlmAgent(
        model="openai/gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        tools=[web_search, web_extract, end_call],
        config=LlmConfig(
            system_prompt=SYSTEM_PROMPT,
            introduction=INTRODUCTION,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=TEMPERATURE,
        ),
    )


app = VoiceAgentApp(get_agent=get_agent)

if __name__ == "__main__":
    app.run()
