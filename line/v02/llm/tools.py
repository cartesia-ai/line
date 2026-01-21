"""
Built-in tools for LLM agents.

Provides web_search tool that:
- Uses native LLM web search when supported (via web_search_options)
- Falls back to DuckDuckGo search for LLMs without native web search

Usage:
    # Default settings
    LlmAgent(tools=[web_search])

    # Custom settings
    LlmAgent(tools=[web_search(search_context_size="high")])
    LlmAgent(tools=[web_search(
        search_context_size="medium",
        user_location={"city": "San Francisco", "country": "US"}
    )])
"""

from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, Literal, Optional, TypedDict

from line.v02.llm.agent import ToolEnv


class UserLocation(TypedDict, total=False):
    """User location for web search."""

    city: str
    region: str
    country: str
    timezone: str


@dataclass
class WebSearchTool:
    """
    Web search tool that uses native LLM web search when available,
    or falls back to DuckDuckGo for unsupported LLMs.

    This class is both:
    1. A marker that LlmAgent detects to enable native web search on supported models
    2. A callable tool that performs actual web search on unsupported models

    Usage:
        # Default settings (medium context size)
        LlmAgent(tools=[web_search])

        # Custom settings
        LlmAgent(tools=[web_search(search_context_size="high")])

        # With user location
        LlmAgent(tools=[web_search(
            search_context_size="medium",
            user_location={"city": "San Francisco", "country": "US"}
        )])
    """

    search_context_size: Literal["low", "medium", "high"] = "medium"
    user_location: Optional[UserLocation] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __call__(
        self,
        search_context_size: Literal["low", "medium", "high"] = "medium",
        user_location: Optional[UserLocation] = None,
        **extra: Any,
    ) -> "WebSearchTool":
        """Create a configured WebSearchTool instance.

        Args:
            search_context_size: Amount of search context to include.
                - "low": Fewer results, faster response
                - "medium": Balanced (default)
                - "high": More results, more comprehensive

            user_location: Optional location hint for localized results.
                Example: {"city": "NYC", "country": "US"}

            **extra: Additional provider-specific options.

        Returns:
            A new WebSearchTool instance with the specified configuration.
        """
        return WebSearchTool(
            search_context_size=search_context_size,
            user_location=user_location,
            extra=extra,
        )

    def get_web_search_options(self) -> Dict[str, Any]:
        """Get the web_search_options dict for litellm.

        Returns a dict suitable for passing as `web_search_options` to litellm's
        completion/chat methods for models that support native web search.
        """
        options: Dict[str, Any] = {
            "search_context_size": self.search_context_size,
        }
        if self.user_location:
            # Transform to litellm's expected format
            options["user_location"] = {
                "type": "approximate",
                "approximate": self.user_location,
            }
        options.update(self.extra)
        return options

    async def search(
        self,
        ctx: ToolEnv,
        query: Annotated[str, "The search query to look up on the web"],
    ) -> str:
        """
        Perform a web search using DuckDuckGo.

        This method is called as a fallback when the LLM doesn't support native
        web search. It uses the duckduckgo-search library to fetch real-time
        web results.

        Args:
            ctx: Tool execution context (unused but required by tool signature).
            query: The search query string.

        Returns:
            Formatted search results as a string, or an error message.
        """
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return (
                "Error: duckduckgo-search package not installed. "
                "Install with: pip install duckduckgo-search"
            )

        try:
            with DDGS() as ddgs:
                # Map context size to number of results
                num_results = {"low": 3, "medium": 5, "high": 10}.get(
                    self.search_context_size, 5
                )

                results = list(ddgs.text(query, max_results=num_results))

                if not results:
                    return f"No results found for: {query}"

                # Format results for LLM consumption
                formatted = []
                for i, r in enumerate(results, 1):
                    title = r.get("title", "")
                    body = r.get("body", "")
                    href = r.get("href", "")
                    formatted.append(f"{i}. {title}\n   {body}\n   URL: {href}")

                return "\n\n".join(formatted)

        except Exception as e:
            return f"Web search error: {e}"


# Default instance - can be used directly or called to configure
# Usage: web_search or web_search(search_context_size="high")
web_search = WebSearchTool()
