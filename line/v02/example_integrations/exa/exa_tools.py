"""Exa web search tool for the Line v0.2 SDK."""

import asyncio
import os
from typing import Annotated

from exa_py import Exa
from loguru import logger

from line.v02.llm import ToolEnv, loopback_tool


@loopback_tool(
    name="web_search",
    description=(
        "Search the web for current information to answer the user's question. "
        "Use this when you need up-to-date facts, statistics, news, or specific "
        "information that requires web research."
    ),
)
async def web_search(
    ctx: ToolEnv,
    query: Annotated[str, "The search query. Be specific and include key terms."],
) -> str:
    """Search the web using Exa API."""
    logger.info(f"Performing Exa web search: '{query}'")

    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        return "Web search failed: EXA_API_KEY not set."

    try:
        client = Exa(api_key=api_key)
        results = await asyncio.to_thread(
            client.search_and_contents,
            query,
            num_results=10,
            type="fast",
            livecrawl="never",
            text={"max_characters": 1000},
        )

        if not results or not results.results:
            return "No relevant information found."

        # Format results for LLM
        content_parts = [f"Search Results for: '{query}'\n"]
        for i, result in enumerate(results.results[:10]):
            content_parts.append(f"\n--- Source {i + 1}: {result.title} ---\n")
            if result.text:
                content_parts.append(f"{result.text}\n")
            content_parts.append(f"URL: {result.url}\n")

        logger.info(f"Search completed: {len(results.results)} sources found")
        return "".join(content_parts)

    except Exception as e:
        logger.error(f"Exa search failed: {e}")
        return f"Web search failed: {e}"
