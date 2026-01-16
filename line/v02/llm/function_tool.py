"""
Function tool definitions for LLM agents.

Provides Field annotation and FunctionTool class for defining tools.
See README.md for examples.
"""

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import Enum
from inspect import Parameter, iscoroutinefunction, signature
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
import warnings


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
class FunctionToolInfo:
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


class FunctionTool:
    """Wrapper that makes a function usable as an LLM tool."""

    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tool_type: ToolType = ToolType.LOOPBACK,
    ):
        self._func = func
        self._name = name or func.__name__
        self._description = description or (func.__doc__ or "").strip()
        self._tool_type = tool_type
        self._parameters = self._extract_parameters()
        self._is_async = iscoroutinefunction(func)

    @property
    def name(self) -> str:
        """Get the tool name."""
        return self._name

    @property
    def description(self) -> str:
        """Get the tool description."""
        return self._description

    @property
    def tool_type(self) -> ToolType:
        """Get the tool type."""
        return self._tool_type

    @property
    def parameters(self) -> Dict[str, ParameterInfo]:
        """Get the tool parameters."""
        return self._parameters

    @property
    def func(self) -> Callable:
        """Get the underlying function."""
        return self._func

    @property
    def is_async(self) -> bool:
        """Check if the function is async."""
        return self._is_async

    def _extract_parameters(self) -> Dict[str, ParameterInfo]:
        """Extract parameter information from the function signature."""
        params = {}
        sig = signature(self._func)

        # Get type hints, handling forward references
        try:
            hints = get_type_hints(self._func, include_extras=True)
        except Exception:
            hints = {}

        for param_name, param in sig.parameters.items():
            # Skip 'self', 'cls', and context parameters
            if param_name in ("self", "cls", "ctx", "context"):
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

    async def __call__(self, ctx: Any, **kwargs) -> Any:
        """Call the underlying function."""
        if self._is_async:
            return await self._func(ctx, **kwargs)
        else:
            return self._func(ctx, **kwargs)

    def get_info(self) -> FunctionToolInfo:
        """Get the tool information."""
        return FunctionToolInfo(
            name=self._name,
            description=self._description,
            func=self._func,
            parameters=self._parameters,
            tool_type=self._tool_type,
        )


def function_tool(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Union[FunctionTool, Callable[[Callable], FunctionTool]]:
    """Deprecated: Use @loopback_tool instead."""
    warnings.warn(
        "function_tool is deprecated, use loopback_tool instead",
        DeprecationWarning,
        stacklevel=2,
    )

    def decorator(f: Callable) -> FunctionTool:
        return FunctionTool(f, name=name, description=description)

    if func is not None:
        return decorator(func)

    return decorator
