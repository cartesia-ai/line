"""
Tool utilities for LLM agents.

This module consolidates:
- FunctionTool class and ParameterInfo for defining tools with Annotated syntax
- ToolType enum for tool execution paradigms
- ToolEnv context class passed to tool functions
- Protocol classes for tool function signatures

Parameter syntax:
    # Required parameter with description
    def my_tool(ctx: ToolEnv, arg: Annotated[str, "this field is required"]):
        ...

    # Optional parameter with default value (required is based on presence of default, not Optional type)
    def my_tool(ctx: ToolEnv, arg: Annotated[str, "has a default"] = "default"):
        ...

    # Optional[T] is unwrapped to T but does NOT make param optional - use a default for that
    def my_tool(ctx: ToolEnv, arg: Annotated[Optional[str], "still required, allows None"]):
        ...

    # Enum constraint using Literal
    def my_tool(ctx: ToolEnv, category: Annotated[Literal["a", "b", "c"], "pick one"]):
        ...
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from inspect import Parameter, signature
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    AsyncIterable,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from line.agent import TurnEnv
from line.events import InputEvent, OutputEvent

if TYPE_CHECKING:
    from line.llm_agent.provider import ParsedModelId
    from line.llm_agent.tools.system import EndCallTool, WebSearchTool

# -------------------------
# Tool Type Enum
# -------------------------


class ToolType(Enum):
    """Tool execution paradigm type."""

    LOOPBACK = "loopback"
    PASSTHROUGH = "passthrough"
    HANDOFF = "handoff"


# -------------------------
# Tool Context
# -------------------------


@dataclass
class ToolEnv:
    """Context passed to tool functions."""

    turn_env: TurnEnv


# -------------------------
# Tool Function Protocols
# -------------------------


class LoopbackToolFn(Protocol):
    """Loopback tool: result is sent back to the LLM for continued generation.

    Signature: (ctx: ToolEnv, **kwargs) -> AsyncIterable[Any] | Awaitable[Any] | Any
    """

    def __call__(self, ctx: ToolEnv, /, **kwargs: Any) -> Union[AsyncIterable[Any], Awaitable[Any], Any]: ...


class PassthroughToolFn(Protocol):
    """Passthrough tool: response bypasses the LLM and goes directly to the user.

    Signature: (ctx: ToolEnv, **kwargs) ->
        AsyncIterable[OutputEvent] | Awaitable[OutputEvent] | OutputEvent
    """

    def __call__(
        self, ctx: ToolEnv, /, **kwargs: Any
    ) -> Union[AsyncIterable[OutputEvent], Awaitable[OutputEvent], OutputEvent]: ...


class HandoffToolFn(Protocol):
    """Handoff tool: transfers control to another agent.

    Signature: (ctx: ToolEnv, event: InputEvent, **kwargs) ->
        AsyncIterable[OutputEvent] | Awaitable[OutputEvent] | OutputEvent

    The event parameter receives AgentHandedOff on initial handoff,
    then subsequent InputEvents for continued processing.
    """

    def __call__(
        self, ctx: ToolEnv, /, event: InputEvent, **kwargs: Any
    ) -> Union[AsyncIterable[OutputEvent], Awaitable[OutputEvent], OutputEvent]: ...


# -------------------------
# Function Tool Definitions
# -------------------------


@dataclass
class FunctionTool:
    """Information about a function tool."""

    name: str
    description: str
    func: Callable
    parameters: Dict[str, "ParameterInfo"]
    tool_type: ToolType = ToolType.LOOPBACK
    is_background: bool = False


# Type alias for tools that can be passed to LlmAgent/LlmProvider.
# Plain callables are automatically wrapped as loopback tools.
# Uses string literal because WebSearchTool/EndCallTool are TYPE_CHECKING-only imports.
ToolSpec = Union[FunctionTool, "WebSearchTool", "EndCallTool", Callable]


@dataclass
class ParameterInfo:
    """Information about a function parameter."""

    name: str
    type_annotation: Type
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None


def _get_tool_name(tool: Any) -> str:
    """Extract the name from a tool (FunctionTool, WebSearchTool, EndCallTool, or Callable)."""
    if hasattr(tool, "name"):
        return tool.name
    if hasattr(tool, "__name__"):
        return tool.__name__
    # Fallback for class instances without a name attr (e.g. WebSearchTool)
    return type(tool).__name__


def _merge_tools(
    base_tools: Optional[List[ToolSpec]], override_tools: Optional[List[ToolSpec]]
) -> List[ToolSpec]:
    """Merge two tool lists, with override_tools replacing base_tools by name."""
    override_names = {_get_tool_name(t) for t in (override_tools or [])}
    filtered_base = [t for t in (base_tools or []) if _get_tool_name(t) not in override_names]
    return filtered_base + (override_tools or [])


# -------------------------
# Parameter Extraction Helpers
# -------------------------


def _is_optional_type(type_hint: Type) -> bool:
    """Check if a type is Optional[X] (i.e., Union[X, None])."""
    origin = get_origin(type_hint)
    if origin is Union:
        args = get_args(type_hint)
        return type(None) in args
    return False


def _unwrap_optional(type_hint: Type) -> Type:
    """Unwrap Optional[X] to X."""
    if _is_optional_type(type_hint):
        args = get_args(type_hint)
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            return non_none_args[0]
    return type_hint


def _extract_parameters(func: Callable) -> Dict[str, ParameterInfo]:
    """Extract parameter information from the function signature.

    Supports:
        - Annotated[type, "description"] for parameters with descriptions
        - param: type = default for optional parameters (required is based on default presence)
        - Optional[type] is unwrapped to type but does NOT affect required status
        - Annotated[Literal["a", "b"], "description"] for enum constraints
    """
    params = {}
    sig = signature(func)

    # Get type hints, handling forward references
    try:
        hints = get_type_hints(func, include_extras=True)
    except Exception:
        hints = {}

    for param_name, param in sig.parameters.items():
        # Skip 'self', 'cls', context parameters, and 'event' (for handoff tools)
        if param_name in ("self", "cls", "ctx", "context", "event"):
            continue

        # Get the type annotation
        type_hint = hints.get(param_name, param.annotation)
        if type_hint is Parameter.empty:
            type_hint = str  # Default to string

        # Check if it's an Annotated type
        description = ""
        actual_type = type_hint

        if get_origin(type_hint) is Annotated:
            args = get_args(type_hint)
            actual_type = args[0]
            # Look for a string description in the annotation
            for arg in args[1:]:
                if isinstance(arg, str):
                    description = arg
                    break

        # Check if the actual type is Optional[X]
        is_optional_type = _is_optional_type(actual_type)
        if is_optional_type:
            actual_type = _unwrap_optional(actual_type)

        # Determine if required based solely on presence of default value
        has_default = param.default is not Parameter.empty
        required = not has_default
        default_value = param.default if has_default else None

        params[param_name] = ParameterInfo(
            name=param_name,
            type_annotation=actual_type,
            description=description,
            required=required,
            default=default_value,
            enum=None,  # Enum handling moved to schema_converter via Literal types
        )

    return params


def _validate_tool_signature(func: Callable, tool_type: ToolType) -> None:
    """Validate that a tool function has the correct signature."""
    sig = signature(func)
    param_names = list(sig.parameters.keys())

    # All tools must have ctx/context as first parameter
    if not param_names:
        raise TypeError(
            f"Tool '{func.__name__}' must have 'ctx' as first parameter. "
            f"Signature: (ctx: ToolEnv, ...) -> ..."
        )

    first_param = param_names[0]
    if first_param not in ("ctx", "context", "self"):
        # Allow 'self' for method-based tools
        raise TypeError(
            f"Tool '{func.__name__}' must have 'ctx' or 'context' as first parameter, "
            f"got '{first_param}'. Signature: (ctx: ToolEnv, ...) -> ..."
        )

    # Handoff tools must have 'event' parameter
    if tool_type == ToolType.HANDOFF:
        if "event" not in param_names:
            raise TypeError(
                f"Handoff tool '{func.__name__}' must have 'event' parameter. "
                f"Signature: (ctx: ToolEnv, ..., event: InputEvent) -> ..."
            )


def construct_function_tool(func, name, description, tool_type, is_background=False):
    """Construct a FunctionTool from a function."""
    _validate_tool_signature(func, tool_type)
    parameters = _extract_parameters(func)
    return FunctionTool(
        name=name or func.__name__,
        description=description or (func.__doc__ or "").strip(),
        func=func,
        parameters=parameters,
        tool_type=tool_type,
        is_background=is_background,
    )


def _normalize_tools(
    tool_specs: List[ToolSpec],
    model_id: ParsedModelId,
) -> Tuple[List[FunctionTool], Optional[Dict[str, Any]]]:
    """Resolve tool specs into FunctionTools and optional web_search_options.

    Converts any tool spec to a FunctionTool:
    - FunctionTool → pass through
    - EndCallTool → .as_function_tool()
    - WebSearchTool → native web search (if model supports it and no other tools)
                      or fallback DuckDuckGo FunctionTool
    - Callable → loopback_tool(callable)

    Uses lazy imports to avoid circular dependencies.

    Args:
        tool_specs: List of tools (FunctionTool, EndCallTool, WebSearchTool, or callable).
        model_id: Parsed model identifier, used for native web search support detection.

    Returns:
        (function_tools, web_search_options) — web_search_options is set only
        when the model supports native web search and there are no other
        function tools; otherwise WebSearchTool is converted to a fallback
        FunctionTool in the first list.
    """
    from line.llm_agent.tools.decorators import loopback_tool
    from line.llm_agent.tools.system import EndCallTool, WebSearchTool

    function_tools: List[FunctionTool] = []
    web_search_tool: Optional[Any] = None

    for tool in tool_specs:
        if isinstance(tool, FunctionTool):
            function_tools.append(tool)
        elif isinstance(tool, EndCallTool):
            function_tools.append(tool.as_function_tool())
        elif isinstance(tool, WebSearchTool):
            web_search_tool = tool
        elif callable(tool):
            function_tools.append(loopback_tool(tool))
        else:
            raise TypeError(
                f"Unsupported tool type: {type(tool).__name__}. "
                f"Expected FunctionTool, EndCallTool, WebSearchTool, or callable."
            )

    web_search_options: Optional[Dict[str, Any]] = None
    if web_search_tool is not None:
        if _check_web_search_support(model_id) and not function_tools:
            web_search_options = web_search_tool.get_web_search_options()
        else:
            function_tools.append(_web_search_tool_to_function_tool(web_search_tool))

    return function_tools, web_search_options


def _check_web_search_support(model_id: ParsedModelId) -> bool:
    """Check if a model supports native web search via litellm.

    Returns True if the model supports web_search_options, False otherwise.
    """
    if model_id.provider == "openai" and model_id.model == "gpt-4.1":
        # LiteLLM thinks 4.1 supports this, but it doesn't
        return False
    try:
        import litellm

        return litellm.supports_web_search(model=str(model_id))
    except (ImportError, AttributeError, Exception):
        # If litellm doesn't have supports_web_search or any error occurs,
        # fall back to the tool-based approach
        return False


def _web_search_tool_to_function_tool(web_search_tool: Any) -> FunctionTool:
    """Convert a WebSearchTool to a FunctionTool for use as a fallback."""
    return construct_function_tool(
        func=web_search_tool.search,
        name="web_search",
        description="Search the web for real-time information."
        + " Use this when you need current information that may not be in your training data.",
        tool_type=ToolType.LOOPBACK,
    )


__all__ = [
    # Tool context
    "ToolEnv",
    # Tool function protocols
    "ToolType",
    "LoopbackToolFn",
    "PassthroughToolFn",
    "HandoffToolFn",
    "FunctionTool",
    "ParameterInfo",
    # constructor
    "construct_function_tool",
    # resolution
    "_normalize_tools",
]
