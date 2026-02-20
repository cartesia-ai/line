"""
Built-in system tools for LLM agents.

Provides end_call, send_dtmf, transfer_call, and web_search tools.
"""

from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, Literal, Optional

from line.agent import Agent
from line.events import (
    AgentEndCall,
    AgentHandedOff,
    AgentSendDtmf,
    AgentSendText,
    AgentTransferCall,
    AgentUpdateCall,
    CallStarted,
)
from line.llm_agent.tools.decorators import passthrough_tool
from line.llm_agent.tools.utils import FunctionTool, ToolEnv, ToolType, construct_function_tool

# Valid DTMF buttons
DtmfButton = Literal["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "*", "#"]


@dataclass
class UpdateCallConfig:
    """Configuration for updating call settings during handoff.

    Used with agent_as_handoff to change voice or other settings when
    transferring to a new agent.
    """

    voice_id: Optional[str] = None
    pronunciation_dict_id: Optional[str] = None


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


class EndCallTool:
    """
    Configurable end_call tool with eagerness levels.

    Controls how readily the LLM will end calls:
    - "low": Very cautious, confirms multiple times before ending
    - "normal": Standard behavior, ends when conversation is complete
    - "high": Ends promptly when user indicates they're done

    Usage:
        # Default (normal eagerness)
        LlmAgent(tools=[end_call])

        # Custom eagerness
        LlmAgent(tools=[end_call(eagerness="low")])

        # Fully custom description
        LlmAgent(tools=[end_call(description="Only end after user says 'goodbye'")])
    """

    _DESCRIPTIONS: Dict[str, str] = {
        "low": (
            "End the call. Before ending, you MUST first ask 'Is there anything else I can help you with?' "
            "and wait for the user to explicitly confirm they have no more questions. "
            "Even if the user says goodbye, ask if there's anything else first. Never assume the conversation is over."
        ),
        "normal": (
            "End the call when the user says goodbye, thanks you, or confirms they're done. "
            "Say goodbye before calling."
        ),
        "high": (
            "End the call promptly when the user indicates they're done or says goodbye. "
            "Say goodbye before calling. Don't ask follow-up questions like 'Is there anything else?'"
        ),
    }

    def __init__(
        self,
        eagerness: Literal["low", "normal", "high"] = "normal",
        description: Optional[str] = None,
    ):
        self.eagerness = eagerness
        self.description = description if description else self._DESCRIPTIONS[eagerness]
        self._function_tool = self._create_function_tool()

    def _create_function_tool(self) -> FunctionTool:
        """Create the underlying FunctionTool with the configured description."""

        async def _end_call_impl(ctx: ToolEnv):
            yield AgentEndCall()

        return construct_function_tool(
            _end_call_impl,
            name="end_call",
            description=self.description,
            tool_type=ToolType.PASSTHROUGH,
        )

    def as_function_tool(self) -> FunctionTool:
        """Return the underlying FunctionTool for use in tool resolution."""
        return self._function_tool

    def __call__(
        self,
        eagerness: Literal["low", "normal", "high"] = "normal",
        description: Optional[str] = None,
    ) -> "EndCallTool":
        """Create a configured EndCallTool instance.

        Args:
            eagerness: How readily the agent should end calls.
                - "low": Very cautious, multiple confirmations required
                - "normal": Standard behavior (default)
                - "high": Ends promptly when user seems done

            description: Optional custom description that overrides eagerness-based text.

        Returns:
            A new EndCallTool instance with the specified configuration.
        """
        return EndCallTool(eagerness=eagerness, description=description)


# Default instance - can be used directly or called to configure
# Usage: end_call or end_call(eagerness="low")
end_call = EndCallTool()


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


def agent_as_handoff(
    agent: Agent,
    *,
    handoff_message: Optional[str] = None,
    update_call: Optional[UpdateCallConfig] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> FunctionTool:
    """
    Create a handoff tool from an Agent.

    This helper wraps an Agent (callable or class with process method) as a handoff tool,
    handling the common pattern of announcing a transfer and delegating events to the agent.

    Args:
        agent: The agent to hand off to. Can be an AgentCallable or AgentClass.
        handoff_message: Optional message to send before handoff (e.g., "Transferring you now...").
        update_call: Optional config to update call settings (voice, pronunciation) before handoff.
        name: Tool name for LLM function calling. Defaults to agent class name or "transfer_to_agent".
        description: Tool description. Defaults to a generic handoff description.

    Returns:
        A FunctionTool that can be passed to LlmAgent's tools list.

    Example:
        spanish_agent = LlmAgent(
            model="gemini/gemini-2.5-flash-preview-09-2025",
            config=LlmConfig(system_prompt="You speak only in Spanish."),
        )

        main_agent = LlmAgent(
            model="gemini/gemini-2.5-flash-preview-09-2025",
            tools=[
                agent_as_handoff(
                    spanish_agent,
                    handoff_message="Transferring you to our Spanish-speaking agent...",
                    update_call=UpdateCallConfig(voice_id="spanish-voice-id"),
                    name="transfer_to_spanish",
                    description="Transfer to a Spanish-speaking agent when requested.",
                ),
            ],
        )
    """

    # Determine tool name
    if name is None:
        if hasattr(agent, "__class__") and agent.__class__.__name__ != "function":
            name = f"transfer_to_{agent.__class__.__name__.lower()}"
        else:
            name = "transfer_to_agent"

    # Determine description
    if description is None:
        description = "Transfer the conversation to another agent."

    async def _handoff_fn(ctx: ToolEnv, event):
        if isinstance(event, AgentHandedOff):
            # Send handoff message if provided
            if handoff_message:
                yield AgentSendText(text=handoff_message)

            # Update call settings (e.g., voice) if provided
            if update_call is not None:
                yield AgentUpdateCall(
                    voice_id=update_call.voice_id,
                    pronunciation_dict_id=update_call.pronunciation_dict_id,
                )

            # Trigger the agent's introduction via CallStarted
            async for output in _call_agent(agent, ctx.turn_env, CallStarted()):
                yield output
            return

        # Delegate subsequent events to the agent
        async for output in _call_agent(agent, ctx.turn_env, event):
            yield output

    # Use construct_function_tool to create the FunctionTool
    return construct_function_tool(
        _handoff_fn,
        name=name,
        description=description,
        tool_type=ToolType.HANDOFF,
    )


def _call_agent(agent: Agent, turn_env, event):
    """Call an agent, handling both AgentClass and AgentCallable."""
    if hasattr(agent, "process"):
        # AgentClass with process method
        return agent.process(turn_env, event)
    else:
        # AgentCallable
        return agent(turn_env, event)


__all__ = [
    "DtmfButton",
    "EndCallTool",
    "UpdateCallConfig",
    "WebSearchTool",
    "web_search",
    "end_call",
    "send_dtmf",
    "transfer_call",
    "agent_as_handoff",
]
