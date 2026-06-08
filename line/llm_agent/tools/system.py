"""
Built-in system tools for LLM agents.

Provides end_call, send_dtmf, transfer_call, and web_search tools.
"""

import asyncio
from dataclasses import dataclass, field
import json
import logging
from typing import Annotated, Any, Dict, Literal, Optional
from urllib.parse import quote as _url_quote

import aiohttp

from line.agent import Agent, call_agent
from line.events import (
    AgentEndCall,
    AgentHandedOff,
    AgentSendDtmf,
    AgentSendText,
    AgentTransferCall,
    AgentUpdateCall,
    CallStarted,
)
from line.knowledge_base import DEFAULT_TOP_K, KnowledgeBaseError, _warn_if_long_timeout
from line.llm_agent.tools import http_server_tool_utils
from line.llm_agent.tools.decorators import passthrough_tool
from line.llm_agent.tools.utils import FunctionTool, ParameterInfo, ToolEnv, ToolType, construct_function_tool

# Valid DTMF buttons
DtmfButton = Literal["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "*", "#"]

# Logger for system tools
logger = logging.getLogger(__name__)

# Sentinel for chainable ``__call__`` configs: distinguishes "argument omitted
# (inherit the current value)" from "explicitly passed None". Needed for
# ``active_turns``, where ``None`` is a real value meaning "never auto-remove".
_INHERIT: Any = object()


@dataclass
class UpdateCallConfig:
    """Configuration for updating call settings during handoff.

    Used with agent_as_handoff to change voice or other settings when
    transferring to a new agent.
    """

    voice_id: Optional[str] = None
    pronunciation_dict_id: Optional[str] = None
    language: Optional[str] = None


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

    name: str = "web_search"
    search_context_size: Literal["low", "medium", "high"] = "medium"
    extra: Dict[str, Any] = field(default_factory=dict)

    def __call__(
        self,
        search_context_size: Optional[Literal["low", "medium", "high"]] = None,
        **extra: Any,
    ) -> "WebSearchTool":
        """Create a configured WebSearchTool instance.

        Unset arguments inherit from this instance, so re-configuring a tool
        doesn't silently reset prior settings.

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
            name=self.name,
            search_context_size=(
                search_context_size if search_context_size is not None else self.search_context_size
            ),
            extra=extra if extra else self.extra,
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
    Configurable end_call tool with custom description.

    The LLM must provide a reason when calling this tool, which encourages
    deliberate decision-making before ending the call.

    Args:
        description: Description that replaces the default. Use this to customize
            when the LLM should end the call.

    Usage:
        # Default behavior (conservative - only ends on explicit goodbye)
        LlmAgent(tools=[end_call])

        # Custom description
        LlmAgent(tools=[end_call(description="End when the customer confirms their order.")])
    """

    DEFAULT_DESCRIPTION = """Ends the conversation and hangs up.

Use when:
- User says goodbye or signals they're finished
- ONLY after you have naturally said goodbye to the user

Don't use when:
- User requests to hold or pause
- User wants to transfer to someone else
- User's intent isn't clear

Note: This action is irreversible. Once invoked, the call terminates and no further communication
is possible."""

    def __init__(
        self,
        description: Optional[str] = None,
        interruptible: bool = True,
        active_turns: Optional[int] = None,
    ):
        self.description = description if description else self.DEFAULT_DESCRIPTION
        self.interruptible = interruptible
        # How many user turns this tool stays available before LlmAgent drops it.
        # None (default) = available for the whole call.
        self.active_turns = active_turns
        self._function_tool = self._create_function_tool()

    @property
    def name(self) -> str:
        """Return the tool name."""
        return "end_call"

    def _create_function_tool(self) -> FunctionTool:
        """Create the underlying FunctionTool with the configured description."""

        async def _end_call_impl(
            ctx: ToolEnv,
            reason: Annotated[str, "The reason for ending the call"],
        ):
            # end_call always reports a normal agent hangup reason.
            yield AgentEndCall(reason="agent_ended", interruptible=self.interruptible)

        ft = construct_function_tool(
            _end_call_impl,
            name="end_call",
            description=self.description,
            tool_type=ToolType.GENERAL,
        )
        ft.active_turns = self.active_turns
        return ft

    def as_function_tool(self) -> FunctionTool:
        """Return the underlying FunctionTool for use in tool resolution."""
        return self._function_tool

    def __call__(
        self,
        description: Optional[str] = None,
        interruptible: Optional[bool] = None,
        active_turns: Any = _INHERIT,
    ) -> "EndCallTool":
        """Create a configured EndCallTool instance.

        Unset arguments inherit from this instance, so re-configuring a tool
        doesn't silently reset prior settings.

        Args:
            description: Description that replaces the default. Use this to customize
                when the LLM should end the call.
            interruptible: Whether the end_call tool is interruptible.
            active_turns: User turns the tool stays available before the agent drops it
                (``None`` = whole call). Omit to inherit the current value.
        Returns:
            A new EndCallTool instance with the specified configuration.
        """
        return EndCallTool(
            description=description if description is not None else self.description,
            interruptible=interruptible if interruptible is not None else self.interruptible,
            active_turns=self.active_turns if active_turns is _INHERIT else active_turns,
        )


# Default instance - can be used directly or called to configure
# Examples:
#   end_call                                              # Use default behavior
#   end_call(description="Only end after explicit goodbye")  # Custom instructions
end_call = EndCallTool()


class TransferCallTool:
    """
    transfer_call tool.

    The destination can be supplied in one of two ways:

    - **Dynamic** (default): the LLM provides ``target_phone_number`` at call
      time. It is validated and normalized to E.164 on each call.
    - **Pinned**: pass ``target_phone_number`` at construction
      (``transfer_call(target_phone_number="+14155551234")``). The number is
      validated once at construction and hidden from the LLM, which then only
      decides *whether* to transfer. This matches the common "escalate to a
      fixed support/human line" pattern.

    The optional fixed ``message`` and ``interruptible`` flag are set at
    construction in both modes, not by the LLM at tool-call time.
    """

    DYNAMIC_DESCRIPTION = (
        "Transfer the call to another phone number (E.164).\n"
        "Do not speak a separate line before calling this tool.\n"
        "If the tool was configured with a fixed message at construction,\n"
        "that message is spoken before the transfer."
    )

    FIXED_DESCRIPTION = (
        "Transfer the call to the preconfigured destination.\n"
        "Use this when the caller should be handed off (for example, when they "
        "ask for a human, a supervisor, or to escalate).\n"
        "Do not speak a separate line before calling this tool.\n"
        "If the tool was configured with a fixed message at construction,\n"
        "that message is spoken before the transfer."
    )

    def __init__(
        self,
        target_phone_number: Optional[str] = None,
        message: Optional[str] = None,
        interruptible: bool = True,
        active_turns: Optional[int] = None,
    ):
        # When pinned, validate/normalize once at construction so a bad config
        # fails fast rather than at call time.
        self.target_phone_number = (
            self._normalize_number(target_phone_number) if target_phone_number is not None else None
        )
        self.message = message
        self.interruptible = interruptible
        # User turns this tool stays available before LlmAgent drops it (None = whole call).
        self.active_turns = active_turns
        self._function_tool = self._create_function_tool()

    @property
    def name(self) -> str:
        """Return the tool name."""
        return "transfer_call"

    @staticmethod
    def _normalize_number(number: str) -> str:
        """Validate and normalize a phone number to E.164, raising on invalid input."""
        import phonenumbers

        try:
            parsed = phonenumbers.parse(number)
        except phonenumbers.NumberParseException as e:
            raise ValueError(
                f"TransferCallTool: could not parse target_phone_number {number!r}: {e}"
            ) from None
        if not phonenumbers.is_valid_number(parsed):
            raise ValueError(f"TransferCallTool: target_phone_number {number!r} is not a valid phone number.")
        return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)

    def _create_function_tool(self) -> FunctionTool:
        """Create the underlying FunctionTool for the configured mode."""
        ft = (
            self._create_fixed_number_tool()
            if self.target_phone_number is not None
            else self._create_dynamic_number_tool()
        )
        ft.active_turns = self.active_turns
        return ft

    def _create_fixed_number_tool(self) -> FunctionTool:
        """Pinned destination: no LLM-facing parameter; transfer to the preset number."""
        fixed_number = self.target_phone_number

        async def _transfer_call_fixed_impl(ctx: ToolEnv):
            """Transfer the call to the preconfigured destination number.

            Exposes no LLM-facing parameter: the destination is pinned at
            construction and hidden from the model, so the schema carries no
            ``target_phone_number`` for it to supply.
            """
            if self.message:
                yield AgentSendText(text=self.message, interruptible=self.interruptible)
            yield AgentTransferCall(target_phone_number=fixed_number, interruptible=self.interruptible)

        return construct_function_tool(
            _transfer_call_fixed_impl,
            name="transfer_call",
            description=self.FIXED_DESCRIPTION,
            tool_type=ToolType.GENERAL,
        )

    def _create_dynamic_number_tool(self) -> FunctionTool:
        """Dynamic destination: the LLM supplies and we validate the number per call."""

        async def _transfer_call_impl(
            ctx: ToolEnv,
            target_phone_number: Annotated[
                str, "The destination phone number in E.164 format (e.g., +14155551234)"
            ],
        ):
            """Transfer the call to another phone number (E.164).
            Do not speak a separate line before calling this tool.
            If the tool was configured with a fixed message at construction,
            that message is spoken before the transfer.
            """
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

            if self.message:
                yield AgentSendText(text=self.message, interruptible=self.interruptible)
            yield AgentTransferCall(target_phone_number=normalized_number, interruptible=self.interruptible)

        return construct_function_tool(
            _transfer_call_impl,
            name="transfer_call",
            description=_transfer_call_impl.__doc__,
            tool_type=ToolType.GENERAL,
        )

    def as_function_tool(self) -> FunctionTool:
        """Return the underlying FunctionTool for use in tool resolution."""
        return self._function_tool

    def __call__(
        self,
        target_phone_number: Optional[str] = None,
        message: Optional[str] = None,
        interruptible: Optional[bool] = None,
        active_turns: Any = _INHERIT,
    ) -> "TransferCallTool":
        """Create a configured TransferCallTool instance.

        Unset arguments inherit from this instance, so a configured tool can be
        refined without silently losing prior config — e.g.
        ``transfer_call(target_phone_number="+1...")(interruptible=False)`` keeps
        the pinned destination.

        Args:
            target_phone_number: Optional pinned destination (E.164). When set,
                the number is hidden from the LLM and validated at construction.
            message: Optional message spoken before transfer.
            interruptible: Whether the transfer_call tool is interruptible.
            active_turns: User turns the tool stays available before the agent drops it
                (``None`` = whole call). Omit to inherit the current value.
        """
        return TransferCallTool(
            target_phone_number=(
                target_phone_number if target_phone_number is not None else self.target_phone_number
            ),
            message=message if message is not None else self.message,
            interruptible=interruptible if interruptible is not None else self.interruptible,
            active_turns=self.active_turns if active_turns is _INHERIT else active_turns,
        )


# Default instance - can be used directly or called to configure
# Examples:
#   transfer_call                                              # LLM supplies the number
#   transfer_call(target_phone_number="+14155551234")          # Pin the destination
#   transfer_call(interruptible=False)                          # Disable interruptibility
transfer_call = TransferCallTool()


class VoicemailTool:
    """
    voicemail tool: the LLM calls this when it detects it has reached a voicemail /
    answering machine. An optional fixed message and interruptibility are set at
    construction (``VoicemailTool(...)`` / ``voicemail(...)``), not by the LLM.

    The detection cues live in the tool's description (below), so the agent's system
    prompt doesn't need to carry voicemail-handling instructions. When invoked the tool
    speaks the configured message (if any) and then ends the call with
    ``reason="voicemail_detected"``, which the Cartesia API records as the call's
    ``end_reason``.

    Behavior modes (all from this one tool):
      - voicemail(message="...")  -> speak the message (uninterruptible), then end.
      - voicemail                 -> silently end the call (no message).
      - dynamic message           -> let the LLM speak in its turn, then call voicemail() to end.
    """

    DEFAULT_DESCRIPTION = """End the call because you've reached a voicemail or answering machine.

Use when the greeting is a machine, e.g.:
- "please leave a message", "at the tone", "you've reached the voicemail of…"
- a beep, or a one-sided recorded greeting with no back-and-forth

Do NOT use this if a real person is talking with you. You may leave a brief message
first; this tool ends the call."""

    def __init__(
        self,
        message: Optional[str] = None,
        interruptible: bool = False,
        description: Optional[str] = None,
        active_turns: Optional[int] = None,
    ):
        # interruptible defaults to False: on a voicemail there's no human to interrupt
        # and we want the message left in full.
        self.message = message
        self.interruptible = interruptible
        self.description = description if description else self.DEFAULT_DESCRIPTION
        # active_turns bounds how long the tool stays available (the agent drops it
        # after that many user turns). Default None = available for the whole call.
        # A finite window is unreliable for voicemail because greetings transcribe
        # into a variable, often large number of short turns ("…forwarded to
        # voicemail.", "…not available.", "At the tone…", "…you may hang up.") that
        # can all arrive before the agent first responds — exhausting a small window
        # so the tool is gone by the time the LLM acts. The tool's description still
        # tells the model not to use it once a real person is talking, so keeping it
        # available is safe; set a finite value to force removal after N turns.
        self.active_turns = active_turns
        self._function_tool = self._create_function_tool()

    @property
    def name(self) -> str:
        """Return the tool name."""
        return "voicemail"

    def _create_function_tool(self) -> FunctionTool:
        """Create the underlying FunctionTool with the configured message and interruptibility."""

        async def _voicemail_impl(ctx: ToolEnv):
            # Keep "voicemail_detected" in sync with EndCallReason in line/events.py.
            if self.message:
                yield AgentSendText(text=self.message, interruptible=self.interruptible)
            yield AgentEndCall(reason="voicemail_detected", interruptible=self.interruptible)

        ft = construct_function_tool(
            _voicemail_impl,
            name="voicemail",
            description=self.description,
            tool_type=ToolType.GENERAL,
        )
        ft.active_turns = self.active_turns
        return ft

    def as_function_tool(self) -> FunctionTool:
        """Return the underlying FunctionTool for use in tool resolution."""
        return self._function_tool

    def __call__(
        self,
        message: Optional[str] = None,
        interruptible: Optional[bool] = None,
        description: Optional[str] = None,
        active_turns: Any = _INHERIT,
    ) -> "VoicemailTool":
        """Create a configured VoicemailTool instance.

        Unset arguments inherit from this instance, so re-configuring a tool
        doesn't silently reset prior settings (e.g.
        ``voicemail(message="…")(active_turns=5)`` keeps the message).

        Args:
            message: Optional message spoken (uninterruptible by default) before the call ends.
            interruptible: Whether the message/end are interruptible.
            description: Override the default LLM-facing description (when to invoke).
            active_turns: User turns the tool stays available before the agent drops it
                (``None`` = whole call, the default). Omit to inherit the current value.
        """
        return VoicemailTool(
            message=message if message is not None else self.message,
            interruptible=interruptible if interruptible is not None else self.interruptible,
            description=description if description is not None else self.description,
            active_turns=self.active_turns if active_turns is _INHERIT else active_turns,
        )


# Default instance - can be used directly or called to configure
# Examples:
#   voicemail                                          # Silently end on voicemail
#   voicemail(message="Hi, please call us back…")      # Leave a message, then end
voicemail = VoicemailTool()


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
        tool_type=ToolType.GENERAL,
    )


class KnowledgeBaseTool:
    """
    knowledge_base tool: lets the LLM look up information from the agent's
    knowledge base by issuing a natural-language query.

    Filters, ``top_k``, and ``timeout_s`` are set at construction
    (``knowledge_base(...)``), not by the LLM at tool-call time. The LLM only
    chooses the query string.

    Usage:
        # Default behavior — no filters
        LlmAgent(tools=[knowledge_base])

        # Pre-filter every retrieval to a specific subset
        LlmAgent(tools=[knowledge_base(filters={"category": "billing"})])

        # Override the per-tool description (e.g. domain-specific guidance)
        LlmAgent(tools=[knowledge_base(description="Look up insurance policy terms.")])

        # Run lookups as a background loopback tool so the LLM can keep
        # talking to the user while the query is in flight.
        LlmAgent(tools=[knowledge_base(is_background=True)])
    """

    DEFAULT_DESCRIPTION = (
        "Look up information from your knowledge base. "
        "Use when the user asks about facts, policies, products, or other domain "
        "information that may be stored in reference documents. "
        "Pass a focused natural-language query describing what you need to find."
        "Always let the user know before calling the tool, since it might take some time to return results."
    )

    CHUNK_SEPARATOR = "\n\n---\n\n"

    def __init__(
        self,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = DEFAULT_TOP_K,
        description: Optional[str] = None,
        timeout_s: Optional[float] = None,
        is_background: bool = False,
        active_turns: Optional[int] = None,
    ):
        self._filters = filters
        self._top_k = top_k
        self._description = description if description else self.DEFAULT_DESCRIPTION
        self._timeout_s = timeout_s
        self._is_background = is_background
        # User turns this tool stays available before LlmAgent drops it (None = whole call).
        self.active_turns = active_turns
        _warn_if_long_timeout(timeout_s, source="KnowledgeBaseTool")
        self._function_tool = self._create_function_tool()

    @property
    def name(self) -> str:
        return "knowledge_base"

    def _create_function_tool(self) -> FunctionTool:
        filters = self._filters
        top_k = self._top_k
        timeout_s = self._timeout_s
        separator = self.CHUNK_SEPARATOR

        async def _knowledge_base_impl(
            ctx: ToolEnv,
            query: Annotated[str, "Natural-language query describing what to look up"],
        ) -> str:
            kb = ctx.knowledge_base()
            try:
                results = await kb.query(
                    query,
                    filters=filters,
                    top_k=top_k,
                    timeout_s=timeout_s,
                )
            except KnowledgeBaseError as e:
                logger.warning("knowledge_base tool error: %s", e)
                return "Knowledge base lookup is currently unavailable."
            chunks = [r["content"] for r in results if r.get("content")]
            if not chunks:
                return "No relevant information found in the knowledge base."
            return separator.join(chunks)

        ft = construct_function_tool(
            _knowledge_base_impl,
            name="knowledge_base",
            description=self._description,
            tool_type=ToolType.GENERAL,
            is_background=self._is_background,
        )
        ft.active_turns = self.active_turns
        return ft

    def as_function_tool(self) -> FunctionTool:
        return self._function_tool

    def __call__(
        self,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        description: Optional[str] = None,
        timeout_s: Optional[float] = None,
        is_background: Optional[bool] = None,
        active_turns: Any = _INHERIT,
    ) -> "KnowledgeBaseTool":
        return KnowledgeBaseTool(
            filters=filters if filters is not None else self._filters,
            top_k=top_k if top_k is not None else self._top_k,
            description=description if description is not None else self._description,
            timeout_s=timeout_s if timeout_s is not None else self._timeout_s,
            is_background=is_background if is_background is not None else self._is_background,
            active_turns=self.active_turns if active_turns is _INHERIT else active_turns,
        )


# Default instance - can be used directly or called to configure
# Examples:
#   knowledge_base                                      # Default lookup, no filters
#   knowledge_base(filters={"category": "billing"})     # Pre-filtered lookup
knowledge_base = KnowledgeBaseTool()


@passthrough_tool
async def send_dtmf(
    ctx: ToolEnv,
    button: Annotated[DtmfButton, "The DTMF button to send (0-9, *, or #)"],
):
    """Send a DTMF tone. Use when the voice system asks you to press a button."""
    yield AgentSendDtmf(button=button)


def agent_as_handoff(
    agent: Agent,
    *,
    handoff_message: Optional[str] = None,
    update_call: Optional[UpdateCallConfig] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    interruptible: bool = True,
) -> FunctionTool:
    """
    Create a handoff tool from an Agent.

    This helper wraps an Agent (callable or class with process method) as a handoff tool,
    handling the common pattern of announcing a transfer and delegating events to the agent.

    Args:
        agent: The agent to hand off to. Can be an AgentCallable or AgentClass.
        interruptible: Whether the transfer is interruptible.
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
                yield AgentSendText(text=handoff_message, interruptible=interruptible)

            # Update call settings (e.g., voice) if provided
            if update_call is not None:
                yield AgentUpdateCall(
                    voice_id=update_call.voice_id,
                    pronunciation_dict_id=update_call.pronunciation_dict_id,
                    language=update_call.language,
                )

            # Trigger the agent's introduction via CallStarted
            async for output in call_agent(agent, ctx.turn_env, CallStarted()):
                yield output
            return

        # Delegate subsequent events to the agent
        async for output in call_agent(agent, ctx.turn_env, event):
            yield output

    # Use construct_function_tool to create the FunctionTool
    return construct_function_tool(
        _handoff_fn,
        name=name,
        description=description,
        tool_type=ToolType.HANDOFF,
    )


def http_server_tool(
    name: str,
    description: str,
    url: str,
    method: str = "POST",
    path_params_schema: Optional[Dict[str, Dict[str, Any]]] = None,
    request_body_schema: Optional[Dict[str, Any]] = None,
    query_params_schema: Optional[Dict[str, Any]] = None,
    auth: Optional[Dict[str, str]] = None,
    headers: Optional[Dict[str, str]] = None,
    content_type: str = "application/json",
    timeout: Optional[float] = None,
    is_background: bool = True,
) -> FunctionTool:
    """Create an HTTP tool that the LLM can call.

    Properties with "constant_value" are hidden from the LLM and injected
    into every request. All inputs are validated at build time.

    See line/llm_agent/README.md for full schema format, examples, and
    response format documentation.

    Args:
        name: Tool name shown to the LLM.
        description: Tool description shown to the LLM.
        url: Request URL. Supports {param} templating.
        method: HTTP method (default "POST").
        path_params_schema: Dict mapping {param} names to property defs
            with "type" and "description". If omitted, path params are
            inferred from the URL as required strings. All path params
            are always required.
        request_body_schema: JSON Schema dict with "type": "object" and
            "properties". Properties in "required" must be filled by the
            LLM; others are optional. Supports nested objects and
            "constant_value" at any depth.
        query_params_schema: Same structure, but scalar types only
            (string, integer, number, boolean).
        auth: Headers with ${ENV_VAR} placeholders resolved from os.environ.
        headers: Additional static headers (must not overlap with auth).
        content_type: Request body content type. "application/json" (default)
            or "application/x-www-form-urlencoded".
        timeout: Request timeout in seconds.
        is_background: Run in shielded background task (default True).

    Returns:
        A FunctionTool for use in LlmAgent(tools=[...]).

    Raises:
        ValueError: On any malformed input (schemas, env vars, collisions).

    Example::

        create_ticket = http_server_tool(
            name="create_ticket",
            description="Creates a support ticket.",
            url="https://example.cartesia.ai/api/{tenant_id}/tickets",
            method="POST",
            request_body_schema={
                "type": "object",
                "required": ["subject"],
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "Short summary of the issue.",
                    },
                    "source": {
                        "type": "string",
                        "constant_value": "voice_agent",
                    },
                },
            },
            auth={"Authorization": "Bearer ${SUPPORT_API_KEY}"},
        )
    """

    # -- 0. Validate inputs & resolve headers ------------------------------------
    path_params = http_server_tool_utils.parse_path_params(name, url)
    http_server_tool_utils.validate_inputs(
        name,
        description,
        url,
        method,
        path_params,
        path_params_schema,
        request_body_schema,
        query_params_schema,
        content_type,
        timeout,
    )
    resolved_headers = http_server_tool_utils.resolve_headers(name, auth, headers)

    # -- 1. Strip constants from schemas ----------------------------------------
    request_body_properties: Dict[str, Any] = {}
    request_body_required: list[str] = []
    constant_values: Dict[str, Any] = {}

    if request_body_schema:
        request_body_properties, request_body_required, constant_values = (
            http_server_tool_utils.strip_constants(
                request_body_schema.get("properties", {}),
                list(request_body_schema.get("required", [])),
            )
        )

    query_properties: Dict[str, Any] = {}
    query_required: list[str] = []
    query_constant_values: Dict[str, Any] = {}

    if query_params_schema:
        query_properties, query_required, query_constant_values = http_server_tool_utils.strip_constants(
            query_params_schema.get("properties", {}),
            list(query_params_schema.get("required", [])),
        )

    # -- 2. Build ParameterInfo for all LLM-visible params ----------------------
    parameters: Dict[str, ParameterInfo] = {}

    all_path_required = list(path_params)  # path params are always required
    for p in path_params:
        prop_def = (path_params_schema or {}).get(
            p, {"type": "string", "description": f"URL path parameter '{p}'"}
        )
        parameters[p] = http_server_tool_utils.build_param_info(p, prop_def, all_path_required)

    for prop_name, prop_def in request_body_properties.items():
        parameters[prop_name] = http_server_tool_utils.build_param_info(
            prop_name, prop_def, request_body_required
        )
    for prop_name, prop_def in query_properties.items():
        parameters[prop_name] = http_server_tool_utils.build_param_info(prop_name, prop_def, query_required)

    # Validate no parameter name collisions across sources
    _seen: Dict[str, str] = {}
    for label, names in [
        ("path", path_params),
        ("body", list(request_body_properties)),
        ("query", list(query_properties)),
    ]:
        for n in names:
            if n in _seen:
                raise http_server_tool_utils.error(
                    name, f"parameter {n!r} appears in both {_seen[n]} and {label}."
                )
            _seen[n] = label

    # -- 3. Build the async implementation --------------------------------------
    upper_method = method.upper()
    request_body_prop_names = set(request_body_properties)
    query_prop_names = set(query_properties)
    path_param_names = set(path_params)
    max_response_chars = 4096

    async def execute_http_request(ctx: ToolEnv, **kwargs: Any) -> str:
        final_url = url
        query_params: Dict[str, Any] = {}
        body: Dict[str, Any] = {}

        missing = [p for p in path_params if p not in kwargs or kwargs[p] is None]
        if missing:
            return json.dumps(
                {
                    "ok": False,
                    "status": None,
                    "error": f"Missing required URL path parameter(s): {', '.join(missing)}.",
                }
            )

        for k, v in kwargs.items():
            if k in path_param_names:
                final_url = final_url.replace(f"{{{k}}}", _url_quote(str(v), safe=""))
            elif k in query_prop_names:
                query_params[k] = v
            elif k in request_body_prop_names:
                body[k] = v

        if query_constant_values:
            query_params.update(query_constant_values)
        if constant_values:
            body = http_server_tool_utils.deep_merge(body, constant_values)

        req_kwargs: Dict[str, Any] = {
            "method": upper_method,
            "url": final_url,
            "headers": resolved_headers or None,
        }
        if body:
            if content_type == "application/x-www-form-urlencoded":
                try:
                    http_server_tool_utils.validate_form_body_values(body)
                except ValueError as exc:
                    return json.dumps({"ok": False, "status": None, "error": str(exc)})
                req_kwargs["data"] = body
            else:
                req_kwargs["json"] = body
        if query_params:
            req_kwargs["params"] = {
                k: str(v).lower() if isinstance(v, bool) else v for k, v in query_params.items()
            }
        if timeout is not None:
            req_kwargs["timeout"] = aiohttp.ClientTimeout(total=timeout)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(**req_kwargs) as resp:
                    text = await resp.text()
                    if len(text) > max_response_chars:
                        text = text[:max_response_chars] + "... (truncated)"
                    result: Dict[str, Any] = {"ok": 200 <= resp.status < 300, "status": resp.status}
                    result["body" if result["ok"] else "error"] = text
                    return json.dumps(result)
        except aiohttp.ClientError as exc:
            return json.dumps(
                {"ok": False, "status": None, "error": f"Request failed: {type(exc).__name__}: {exc}"}
            )
        except asyncio.TimeoutError:
            detail = f" after {timeout}s" if timeout is not None else ""
            return json.dumps({"ok": False, "status": None, "error": f"Request timed out{detail}."})

    # -- 4. Construct FunctionTool directly -------------------------------------
    return FunctionTool(
        name=name,
        description=description,
        func=execute_http_request,
        parameters=parameters,
        tool_type=ToolType.GENERAL,
        is_background=is_background,
    )


__all__ = [
    "DtmfButton",
    "EndCallTool",
    "TransferCallTool",
    "VoicemailTool",
    "UpdateCallConfig",
    "WebSearchTool",
    "web_search",
    "mcp_tool",
    "end_call",
    "send_dtmf",
    "transfer_call",
    "voicemail",
    "agent_as_handoff",
    "http_server_tool",
]
