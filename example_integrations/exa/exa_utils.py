import asyncio
import sys

sys.path.append("../../")  # Add path to line SDK

from typing import Any, Dict, List

from exa_py import Exa

from line.events import (
    AgentResponse,
    ToolResult,
    UserTranscriptionReceived,
)
from line.tools.system_tools import EndCallArgs

# Tool schema for web search
web_search_schema = {
    "type": "function",
    "function": {
        "name": "web_search",
        "strict": True,
        "description": (
            "Search the web for current information to answer the user's question. "
            "Use this when you need up-to-date facts, statistics, news, or specific "
            "information that requires web research."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The search query optimized for finding relevant information. "
                        "Be specific and include key terms."
                    ),
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

# End call schema
end_call_schema = {
    "type": "function",
    "function": {
        "name": "end_call",
        "strict": True,
        "description": (
            "Ends the call when the user says they need to leave or want to stop the conversation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "goodbye_message": {
                    "type": "string",
                    "description": EndCallArgs.model_fields["goodbye_message"].description,
                }
            },
            "required": ["goodbye_message"],
            "additionalProperties": False,
        },
    },
}


class ExaSearchClient:
    """Wrapper for Exa API client with optimized search functionality."""

    def __init__(self, api_key: str):
        self.client = Exa(api_key=api_key)

    async def search_and_get_content(self, query: str) -> Dict[str, Any]:
        """
        Perform web search and return formatted results.

        Args:
            query: Search query

        Returns:
            Dictionary with search results and metadata
        """
        try:
            # Perform search with content using the exact API call format
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
            formatted_results = self._format_search_results(results, query)
            return formatted_results

        except Exception as e:
            return {"error": f"Search failed: {str(e)}", "query": query, "results": []}

    def _format_search_results(self, raw_results, original_query: str) -> Dict[str, Any]:
        """Format raw Exa results into LLM-friendly structure."""

        if not raw_results or not raw_results.results:
            return {
                "query": original_query,
                "results_summary": "No results found for this query.",
                "source_count": 0,
                "sources": [],
                "formatted_content": "No relevant information found.",
            }

        sources = []
        content_parts = []

        for i, result in enumerate(raw_results.results[:10]):  # Use all 10 results
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
                # Use the text content (limited to 1000 chars by API call)
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


def convert_messages_to_openai(messages: List[dict], system_prompt: str) -> List[Dict[str, str]]:
    """
    Convert Line SDK conversation messages to OpenAI format.

    Args:
        messages: List of conversation events from Line SDK
        system_prompt: System prompt for the conversation

    Returns:
        List of OpenAI-formatted messages
    """
    openai_messages = [{"role": "system", "content": system_prompt}]

    for message in messages:
        if isinstance(message, AgentResponse):
            openai_messages.append({"role": "assistant", "content": message.content})
        elif isinstance(message, UserTranscriptionReceived):
            openai_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, ToolResult):
            # Add tool results as system messages
            openai_messages.append(
                {
                    "role": "system",
                    "content": (
                        f"Tool '{message.tool_name}' executed. Result: {message.result_str or 'Success'}"
                    ),
                }
            )
        # Skip other event types for now

    return openai_messages
