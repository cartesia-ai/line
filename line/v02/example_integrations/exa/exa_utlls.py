"""
Exa web search tool for the Line v0.2 SDK.

Provides a loopback tool that searches the web using the Exa API and returns
formatted results to the LLM for synthesis.
"""

import asyncio
import os
from typing import Annotated, Any, Dict, List, Optional

from exa_py import Exa
from loguru import logger

from line.v02.llm import ToolEnv, loopback_tool


class ExaSearchClient:
    """Wrapper for Exa API client with optimized search functionality."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("EXA_API_KEY")
        if not self.api_key:
            raise ValueError("EXA_API_KEY must be provided or set in environment")
        self.client = Exa(api_key=self.api_key)

    async def search_and_get_content(self, query: str) -> Dict[str, Any]:
        """
        Perform web search and return formatted results.

        Args:
            query: Search query

        Returns:
            Dictionary with search results and metadata
        """
        try:
            # Run the synchronous Exa API call in a thread to avoid blocking
            results = await asyncio.to_thread(
                self.client.search_and_contents,
                query,
                num_results=10,
                type="fast",
                livecrawl="never",
                text={"max_characters": 1000},
            )

            # Format results for LLM consumption
            return self._format_search_results(results, query)

        except Exception as e:
            logger.error(f"Exa search failed: {e}")
            return {"error": f"Search failed: {str(e)}", "query": query, "results": []}

    def _format_search_results(self, raw_results: Any, original_query: str) -> Dict[str, Any]:
        """Format raw Exa results into LLM-friendly structure."""

        if not raw_results or not raw_results.results:
            return {
                "query": original_query,
                "results_summary": "No results found for this query.",
                "source_count": 0,
                "sources": [],
                "formatted_content": "No relevant information found.",
            }

        sources: List[Dict[str, Any]] = []
        content_parts: List[str] = []

        for i, result in enumerate(raw_results.results[:10]):
            # Extract key information
            source_info = {
                "title": result.title or "Untitled",
                "url": result.url,
                "published_date": result.published_date,
                "author": result.author,
            }
            sources.append(source_info)

            # Build content summary for this source
            content_part = f"\n--- Source {i + 1}: {result.title} ---\n"

            if result.text:
                content_part += f"Content: {result.text}\n"

            content_part += f"Source: {result.url}\n"
            content_parts.append(content_part)

        # Create comprehensive formatted content
        formatted_content = f"Search Results for: '{original_query}'\n"
        formatted_content += f"Found {len(raw_results.results)} relevant sources\n"
        formatted_content += "".join(content_parts)

        return {
            "query": original_query,
            "results_summary": f"Found {len(raw_results.results)} relevant results",
            "source_count": len(raw_results.results),
            "sources": sources,
            "formatted_content": formatted_content,
        }


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
    query: Annotated[
        str,
        "The search query optimized for finding relevant information. Be specific and include key terms.",
    ],
) -> str:
    """
    Search the web using Exa API.

    This is a loopback tool - results are sent back to the LLM to be
    synthesized into a conversational response.
    """
    logger.info(f"🔍 Performing Exa web search: '{query}'")

    client = ExaSearchClient()
    results = await client.search_and_get_content(query)

    if "error" in results:
        logger.error(f"Search failed: {results['error']}")
        return f"Web search failed: {results['error']}. Please provide an answer based on your existing knowledge."

    logger.info(f"📊 Search completed: {results['source_count']} sources found")

    # Return formatted content for the LLM to process
    return (
        f"{results['formatted_content']}\n\n"
        "Please synthesize this information to answer the user's question. "
        "Cite sources when relevant."
    )
