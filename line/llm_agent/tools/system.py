"""
Built-in system tools for LLM agents.

Provides end_call, send_dtmf, transfer_call, and web_search tools.
"""

from dataclasses import dataclass, field
import logging
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

# Logger for system tools
logger = logging.getLogger(__name__)


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
            "Even if the user says goodbye, ask if there's anything else first. "
            "Never assume the conversation is over."
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
        # Validate eagerness parameter at runtime and default to "normal" if invalid
        if eagerness not in self._DESCRIPTIONS:
            valid_values = ", ".join(f"'{k}'" for k in self._DESCRIPTIONS.keys())
            logger.warning(
                f"Invalid eagerness value '{eagerness}'. Must be one of: {valid_values}. "
                f"Defaulting to 'normal'."
            )
            eagerness = "normal"
        self.eagerness = eagerness
        self.description = description if description else self._DESCRIPTIONS[eagerness]
        self._function_tool = self._create_function_tool()

    @property
    def name(self) -> str:
        """Return the tool name."""
        return "end_call"

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
                If an invalid value is provided, a warning is logged and "normal" is used.

            description: Optional custom description that overrides eagerness-based text.

        Returns:
            A new EndCallTool instance with the specified configuration.
        """
        # Validation happens in __init__
        return EndCallTool(eagerness=eagerness, description=description)


# Default instance - can be used directly or called to configure
# Usage: end_call or end_call(eagerness="low")
end_call = EndCallTool()


@dataclass
class _McpServer:
    """Internal: holds MCP server connection config and the execute method."""

    name: str
    server_url: Optional[str] = None
    server_config: Dict[str, Any] = field(default_factory=dict)

    async def execute(
        self,
        ctx: ToolEnv,
        tool_name: Annotated[
            Optional[str],
            "The MCP tool to call. Omit to list available tools.",
        ] = None,
        tool_args: Annotated[
            Optional[str],
            "JSON-encoded arguments for the tool (when tool_name is provided).",
        ] = None,
    ) -> str:
        """
        List available tools or execute a specific tool on the MCP server.

        The LLM calls this with no arguments to discover tools,
        or with ``tool_name`` and ``tool_args`` to invoke one.
        """
        import json

        # Parse tool_args up front so a bad value doesn't crash inside the MCP session.
        parsed_args: Dict[str, Any] = {}
        if tool_args is not None and tool_name is not None:
            if isinstance(tool_args, dict):
                parsed_args = tool_args
            else:
                try:
                    parsed_args = json.loads(tool_args)
                except (ValueError, json.JSONDecodeError):
                    return json.dumps(
                        {
                            "error": f"tool_args must be a JSON object, got: {tool_args!r}. "
                            "Call with no arguments first to see each tool's expected schema."
                        }
                    )

        try:
            async with self._connect() as session:
                if tool_name is None or tool_name == "list_tools":
                    result = await session.list_tools()
                    tools_list = []
                    for tool in result.tools:
                        info: Dict[str, Any] = {
                            "name": tool.name,
                            "description": tool.description or "",
                        }
                        if tool.inputSchema:
                            info["inputSchema"] = tool.inputSchema
                        tools_list.append(info)
                    return json.dumps({"tools": tools_list}, indent=2)

                result = await session.call_tool(tool_name, arguments=parsed_args)

                if result.isError:
                    return json.dumps({"error": str(result.content)})

                parts = []
                for content in result.content:
                    if hasattr(content, "text"):
                        parts.append(content.text)
                    elif hasattr(content, "data"):
                        parts.append(str(content.data))
                    else:
                        parts.append(str(content))
                return "\n".join(parts) if parts else json.dumps({"result": "success"})

        except ImportError:
            return json.dumps({"error": "MCP SDK not installed (requires Python >= 3.10)."})
        except Exception as e:
            return json.dumps({"error": f"MCP error on server '{self.name}': {e}"})

    # -- internals ----------------------------------------------------------

    def _connect(self):
        """Return an async context manager that yields a ``ClientSession``.

        Chooses transport based on available configuration:
        - ``server_url`` → streamable-HTTP
        - ``command`` in server_config → stdio
        """
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _ctx():
            from mcp import ClientSession

            if self.server_url:
                from mcp.client.streamable_http import streamablehttp_client

                async with streamablehttp_client(self.server_url) as (r, w, _):
                    async with ClientSession(r, w) as session:
                        await session.initialize()
                        yield session
            else:
                # dunno if we need to support this?
                command = self.server_config.get("command")
                if not command:
                    raise ValueError(f"mcp_tool(name='{self.name}'): provide server_url or command.")

                import shlex

                from mcp import StdioServerParameters
                from mcp.client.stdio import stdio_client

                if isinstance(command, str):
                    parts = shlex.split(command)
                else:
                    parts = list(command)
                params = StdioServerParameters(
                    command=parts[0],
                    args=parts[1:],
                    env=self.server_config.get("env"),
                )
                async with stdio_client(params) as (r, w):
                    async with ClientSession(r, w) as session:
                        await session.initialize()
                        yield session

        return _ctx()


def mcp_tool(name: str, server_url: Optional[str] = None, **server_config: Any) -> "FunctionTool":
    """
    Create an MCP tool for accessing Model Context Protocol servers.

    Returns a FunctionTool that the LLM can call to list or invoke tools
    on the MCP server.

    Args:
        name: A label identifying this MCP server (e.g., "github", "dmcp").
        server_url: HTTP/SSE URL for remote MCP servers.
        **server_config: Additional options.
            - command: Shell command to launch a local stdio server.
            - env: Environment variables for the stdio server process.

    Returns:
        A FunctionTool that can be passed to LlmAgent's tools list.

    Example:
        # Remote HTTP server
        LlmAgent(tools=[mcp_tool(
            name="dmcp",
            server_url="https://dmcp-server.deno.dev/sse",
        )])

        # Local stdio server
        LlmAgent(tools=[mcp_tool(
            name="memory",
            command="npx -y @modelcontextprotocol/server-memory",
        )])
    """
    import sys

    if sys.version_info < (3, 10):
        raise ImportError(
            "mcp_tool requires Python >= 3.10 (the 'mcp' package is not supported on older versions)."
        )

    try:
        import mcp as _mcp  # noqa: F401
    except ImportError:
        raise ImportError(
            "mcp_tool requires the 'mcp' package. Install it with: pip install 'mcp>=1.0.0'"
        ) from None

    if not server_url and "command" not in server_config:
        raise ValueError(f"mcp_tool(name='{name}'): provide either server_url or command.")

    from line.llm_agent.tools.utils import ToolType, construct_function_tool

    mcp_server = _McpServer(name=name, server_url=server_url, server_config=server_config)
    return construct_function_tool(
        func=mcp_server.execute,
        name=f"mcp_{name}",
        description=f"Access the '{name}' MCP server. "
        "Call without arguments to list available tools, "
        "or with tool_name and tool_args to invoke one.",
        tool_type=ToolType.LOOPBACK,
    )


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
    "mcp_tool",
    "end_call",
    "send_dtmf",
    "transfer_call",
    "agent_as_handoff",
]
