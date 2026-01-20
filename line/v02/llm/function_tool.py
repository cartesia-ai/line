"""
Function tool definitions for LLM agents.

Provides Field annotation and FunctionTool class for defining tools.
See README.md for examples.
"""

from dataclasses import dataclass
from dataclasses import field as dataclass_field
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
    get_args,
    get_origin,
    get_type_hints,
)


@dataclass
class Field:
    """
    Field annotation for tool parameters. Use with Annotated[type, Field(...)].

    Attributes:
        description: Description shown to the LLM.
        default: Default value (makes parameter optional).
        enum: List of allowed values.
    """

    description: str = ""
    default: Any = dataclass_field(default_factory=lambda: _MISSING)
    enum: Optional[List[Any]] = None


class _MissingSentinel:
    """Sentinel class to indicate a missing value (distinct from None)."""

    pass


_MISSING = _MissingSentinel()


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


@dataclass
class ParameterInfo:
    """Information about a function parameter."""

    name: str
    type_annotation: Type
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None


def _extract_parameters(func: Callable) -> Dict[str, ParameterInfo]:
    """Extract parameter information from the function signature."""
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

        # Check if it's an Annotated type with Field metadata
        field_info = None
        actual_type = type_hint

        if get_origin(type_hint) is Annotated:
            args = get_args(type_hint)
            actual_type = args[0]
            for arg in args[1:]:
                if isinstance(arg, Field):
                    field_info = arg
                    break

        # Determine if required and get default
        required = True
        default_value = None

        if param.default is not Parameter.empty:
            required = False
            default_value = param.default
        elif field_info and not isinstance(field_info.default, _MissingSentinel):
            required = False
            default_value = field_info.default

        # Get description and enum from Field
        description = ""
        enum_values = None

        if field_info:
            description = field_info.description
            enum_values = field_info.enum

        params[param_name] = ParameterInfo(
            name=param_name,
            type_annotation=actual_type,
            description=description,
            required=required,
            default=default_value,
            enum=enum_values,
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
            f"Signature: (ctx: ToolContext, ...) -> ..."
        )

    first_param = param_names[0]
    if first_param not in ("ctx", "context", "self"):
        # Allow 'self' for method-based tools
        raise TypeError(
            f"Tool '{func.__name__}' must have 'ctx' or 'context' as first parameter, "
            f"got '{first_param}'. Signature: (ctx: ToolContext, ...) -> ..."
        )

    # Handoff tools must have 'event' parameter
    if tool_type == ToolType.HANDOFF:
        if "event" not in param_names:
            raise TypeError(
                f"Handoff tool '{func.__name__}' must have 'event' parameter. "
                f"Signature: (ctx: ToolContext, ..., event: InputEvent) -> ..."
            )


def construct_function_tool(func, name, description, tool_type):
    _validate_tool_signature(func, tool_type)
    parameters = _extract_parameters(func)
    return FunctionTool(
        name=name or func.__name__,
        description=description or (func.__doc__ or "").strip(),
        func=func,
        parameters=parameters,
        tool_type=tool_type,
    )
