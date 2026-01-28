"""
Built-in tools for LLM agents.
"""

from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, Literal, Optional

from line.v02.events import AgentEndCall as AgentEndCallEvent
from line.v02.events import AgentSendDtmf, AgentSendText, AgentTransferCall
from line.v02.llm.agent import ToolEnv
from line.v02.llm.tool_types import passthrough_tool

# Valid DTMF buttons
DtmfButton = Literal["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "*", "#"]


@dataclass
class WebSearchTool:
    """
    Web search tool that uses native LLM web search when available,
    or falls back to DuckDuckGo for unsupported LLMs.
    View models supported with LiteLLM at https://docs.litellm.ai/docs/completion/web_search

    This class is both:
    1. A marker that LlmAgent detects to enable native web search on supported models
    2. A callable tool that performs actual web search on unsupported models

    Usage:
        # Default settings (medium context size)
        LlmAgent(tools=[web_search])

        # Custom settings
        LlmAgent(tools=[web_search(search_context_size="high")])
    """

    search_context_size: Literal["low", "medium", "high"] = "medium"
    extra: Dict[str, Any] = field(default_factory=dict)

    def __call__(
        self,
        search_context_size: Literal["low", "medium", "high"] = "medium",
        **extra: Any,
    ) -> "WebSearchTool":
        """Create a configured WebSearchTool instance.

        Args:
            search_context_size: Amount of search context to include.
                - "low": Fewer results, faster response
                - "medium": Balanced (default)
                - "high": More results, more comprehensive

            **extra: Additional provider-specific options.

        Returns:
            A new WebSearchTool instance with the specified configuration.
        """
        return WebSearchTool(
            search_context_size=search_context_size,
            extra=extra,
        )

    def get_web_search_options(self) -> Dict[str, Any]:
        """Get the web_search_options dict for LiteLLM.

        Returns a dict suitable for passing as `web_search_options` to litellm's
        completion/chat methods for models that support native web search.
        """
        options: Dict[str, Any] = {
            "search_context_size": self.search_context_size,
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
            from ddgs import DDGS
        except ImportError:
            return (
                "Error: duckduckgo-search package not installed. Install with: pip install duckduckgo-search"
            )

        try:
            with DDGS() as ddgs:
                # Map context size to number of results
                num_results = {"low": 3, "medium": 5, "high": 10}.get(self.search_context_size, 5)

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


@passthrough_tool
async def end_call(ctx: ToolEnv):
    """End the call. Say goodbye in your response before calling this."""
    yield AgentEndCallEvent()


@passthrough_tool
async def send_dtmf(
    ctx: ToolEnv,
    button: Annotated[DtmfButton, "The DTMF button to send (0-9, *, or #)"],
):
    """Send a DTMF tone. Use when the voice system asks you to press a button."""
    yield AgentSendDtmf(button=button)


@passthrough_tool
async def transfer_call(
    ctx: ToolEnv,
    target_phone_number: Annotated[str, "The destination phone number in E.164 format (e.g., +14155551234)"],
    message: Annotated[Optional[str], "Optional message to say before transferring"] = None,
):
    """Transfer the call to another phone number."""
    import phonenumbers

    try:
        parsed = phonenumbers.parse(target_phone_number)
        if not phonenumbers.is_valid_number(parsed):
            yield AgentSendText(text="I'm sorry, that phone number appears to be invalid.")
            return
    except phonenumbers.NumberParseException:
        yield AgentSendText(text="I'm sorry, I couldn't understand that phone number format.")
        return

    # Normalize to E.164 format
    normalized_number = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)

    if message is not None:
        yield AgentSendText(text=message)
    yield AgentTransferCall(target_phone_number=normalized_number)
