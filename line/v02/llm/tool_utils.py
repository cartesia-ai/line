"""
Function tool definitions for LLM agents.

Provides FunctionTool class for defining tools with simple Annotated syntax.
See README.md for examples.

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

from dataclasses import dataclass
from enum import Enum
from inspect import Parameter, signature
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)


class ToolType(Enum):
    """Tool execution paradigm type."""

    LOOPBACK = "loopback"
    PASSTHROUGH = "passthrough"
    HANDOFF = "handoff"


@dataclass
class FunctionTool:
    """Information about a function tool."""

    name: str
    description: str
    func: Callable
    parameters: Dict[str, "ParameterInfo"]
    tool_type: ToolType = ToolType.LOOPBACK
    is_background: bool = False


@dataclass
class ParameterInfo:
    """Information about a function parameter."""

    name: str
    type_annotation: Type
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None


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
