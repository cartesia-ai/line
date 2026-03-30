"""
Schema converter utilities for converting tools to LiteLLM format.

This module provides functions to convert FunctionTool instances
to the tool format expected by LiteLLM (OpenAI-compatible).

Example:
    ```python
    from typing import Annotated
    from line.llm_agent import loopback_tool
    from line.llm_agent.schema_converter import function_tool_to_litellm

    @loopback_tool
    async def my_tool(ctx, param: Annotated[str, "Parameter description"]):
        '''Tool description'''
        ...

    litellm_tool = function_tool_to_litellm(my_tool)
    ```

TypedDict Support:
    For nested object parameters, use TypedDict to define the structure.
    This generates proper JSON schemas that work with OpenAI's strict mode.

    ```python
    from typing import TypedDict, Annotated

    class MenuItem(TypedDict):
        menu_item_id: str
        quantity: int
        modifiers: list[str]

    @loopback_tool
    async def add_items(ctx, items: Annotated[list[MenuItem], "Items to order"]):
        '''Add items to order'''
        ...
    ```

    Tool schemas must satisfy OpenAI strict rules. Bare ``dict`` / ``list[dict]`` cannot,
    so conversion raises ``ValueError``. Use ``typing.TypedDict`` for nested objects.

    TypedDict with optional fields (``total=False`` or ``NotRequired``) is also
    incompatible with strict mode, so conversion fails. Use only fully-required
    object shapes.
"""

from enum import Enum
from typing import Any, Literal, Optional, Type, Union, get_args, get_origin, get_type_hints

from line.llm_agent.tools.utils import FunctionTool, ParameterInfo


def _is_typeddict(tp: Type) -> bool:
    """Check if a type is a TypedDict.

    TypedDict classes have __required_keys__ and __optional_keys__ attributes.
    """
    return isinstance(tp, type) and hasattr(tp, "__required_keys__") and hasattr(tp, "__optional_keys__")


def python_type_to_json_schema(type_annotation: Type, *, strict: bool = True) -> dict[str, Any]:
    """
    Convert a Python type annotation to a JSON Schema type.

    Args:
        type_annotation: The Python type to convert.
        strict: Whether to add additionalProperties: false to objects (for OpenAI compatibility).

    Returns:
        A dictionary representing the JSON Schema type.
    """
    # Handle None type
    if type_annotation is type(None):
        return {"type": "null"}

    # Handle basic types
    type_map = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
    }

    if type_annotation in type_map:
        return type_map[type_annotation]

    # Handle plain dict - error in strict mode since it cannot satisfy OpenAI strict rules
    if type_annotation is dict:
        if strict:
            raise ValueError(
                "Using 'dict' in tool parameters yields an object schema without explicit "
                "properties, which cannot satisfy OpenAI strict mode. "
                "Use TypedDict to define the expected structure. "
                "See: https://platform.openai.com/docs/guides/structured-outputs"
            )
        return {"type": "object"}

    # Handle TypedDict - generates full schema with properties
    if _is_typeddict(type_annotation):
        # OpenAI strict mode requires additionalProperties: false, which means
        # all properties must be in 'required'. TypedDict with optional keys cannot satisfy this.
        if strict and type_annotation.__optional_keys__:
            raise ValueError(
                f"TypedDict '{type_annotation.__name__}' has optional keys, which cannot satisfy "
                "OpenAI strict mode. Use a TypedDict with all required fields. "
                "See: https://platform.openai.com/docs/guides/structured-outputs"
            )

        properties = {}
        required = []

        # Get type hints for the TypedDict fields
        try:
            hints = get_type_hints(type_annotation)
        except Exception:
            hints = getattr(type_annotation, "__annotations__", {})

        for field_name, field_type in hints.items():
            properties[field_name] = python_type_to_json_schema(field_type, strict=strict)
            # Check if field is required (in __required_keys__)
            if field_name in type_annotation.__required_keys__:
                required.append(field_name)

        schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required
        if strict:
            schema["additionalProperties"] = False
        return schema

    # Handle list[X]
    origin = get_origin(type_annotation)
    args = get_args(type_annotation)

    if origin is list:
        if args:
            item_schema = python_type_to_json_schema(args[0], strict=strict)
            return {"type": "array", "items": item_schema}
        return {"type": "array"}

    # Handle dict[K, V] - error in strict mode since it cannot satisfy OpenAI strict rules
    if origin is dict:
        if strict:
            raise ValueError(
                "Using 'dict[K, V]' in tool parameters yields an object schema without explicit "
                "properties, which cannot satisfy OpenAI strict mode. "
                "Use TypedDict to define the expected structure. "
                "See: https://platform.openai.com/docs/guides/structured-outputs"
            )
        return {"type": "object"}

    # Handle Literal types (e.g., Literal["a", "b", "c"])
    if origin is Literal:
        values = list(args)
        # Infer type from the literal values
        if all(isinstance(v, str) for v in values):
            return {"type": "string", "enum": values}
        elif all(isinstance(v, int) for v in values):
            return {"type": "integer", "enum": values}
        elif all(isinstance(v, bool) for v in values):
            return {"type": "boolean", "enum": values}
        else:
            # Mixed types - just return enum without type
            return {"enum": values}

    # Handle Union types (including Optional)
    if origin is Union:
        # Filter out NoneType for Optional handling
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            # This is Optional[X], just return the schema for X
            return python_type_to_json_schema(non_none_args[0], strict=strict)
        # For true Union types, use anyOf
        return {"anyOf": [python_type_to_json_schema(a, strict=strict) for a in non_none_args]}

    # Handle Enum types
    if isinstance(type_annotation, type) and issubclass(type_annotation, Enum):
        return {"type": "string", "enum": [e.value for e in type_annotation]}

    # Default to string for unknown types
    return {"type": "string"}


def build_parameters_schema(parameters: dict[str, ParameterInfo], *, strict: bool = True) -> dict[str, Any]:
    """
    Build a JSON Schema for function parameters.

    Args:
        parameters: Dictionary of parameter info.
        strict: Whether to generate strict mode compatible schemas
            (adds additionalProperties: false to nested objects).

    Returns:
        A JSON Schema object describing the parameters.
    """
    properties = {}
    required = []

    for name, param in parameters.items():
        prop = python_type_to_json_schema(param.type_annotation, strict=strict)

        if param.description:
            prop["description"] = param.description

        if param.enum:
            prop["enum"] = param.enum

        if param.default is not None and not param.required:
            prop["default"] = param.default

        properties[name] = prop

        if param.required:
            required.append(name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}

    if required:
        schema["required"] = required

    return schema


def function_tool_to_litellm(tool: FunctionTool, *, strict: bool = True) -> dict[str, Any]:
    """
    Convert a FunctionTool to LiteLLM (OpenAI Chat Completions) tool format.

    Args:
        tool: The FunctionTool to convert.
        strict: Whether to require OpenAI-compatible strict schemas (default True).
            When True, conversion raises if the tool schema cannot satisfy strict rules
            (e.g. bare ``dict``, optional-key TypedDict). When False, nested objects omit
            the strict lock and the result does not set ``"strict": true``.

    Returns:
        Tool definition dictionary.

    Example:
        ```python
        @loopback_tool
        async def get_weather(ctx, city: Annotated[str, "City name"]):
            '''Get the weather'''
            ...

        tool = function_tool_to_litellm(get_weather)
        # Returns:
        # {
        #     "type": "function",
        #     "function": {
        #         "name": "get_weather",
        #         "description": "Get the weather",
        #         "parameters": {...},
        #         "strict": True
        #     }s
        # }
        ```
    """
    return {
        "type": "function",
        "function": _build_function_tool_payload(tool, strict=strict),
    }


def function_tool_to_openai(
    tool: FunctionTool,
    *,
    strict: bool = True,
    responses_api: bool = False,
) -> Dict[str, Any]:
    """Convert a FunctionTool to OpenAI/Realtimes API tool format."""
    payload = _build_function_tool_payload(tool, strict=strict)
    if responses_api:
        return {
            "type": "function",
            **payload,
        }
    return {
        "type": "function",
        "function": payload,
    }


def tools_to_litellm(tools: list[FunctionTool], *, strict: bool = True) -> list[dict[str, Any]]:
    """
    Convert a list of FunctionTools to LiteLLM format.

    Args:
        tools: list of FunctionTool instances.
        strict: Whether to enable strict mode for function tools.

    Returns:
        list of tool definitions.
    """
    return [function_tool_to_litellm(t, strict=strict) for t in tools]


def function_tools_to_openai(
    tools: list[FunctionTool],
    *,
    strict: bool = True,
    responses_api: bool = False,
) -> list[dict[str, Any]]:
    """Convert a list of FunctionTools to OpenAI/Realtimes API tool format."""
    return [function_tool_to_openai(t, strict=strict, responses_api=responses_api) for t in tools]


def build_openai_tool_defs(
    tools: Optional[list[FunctionTool]] = None,
    *,
    web_search_options: Optional[dict[str, Any]] = None,
    strict: bool = True,
    responses_api: bool = False,
) -> Optional[list[dict[str, Any]]]:
    """Build OpenAI/Realtimes tool definitions including native web search.

    WebSocket-mode backends do not use LiteLLM's ``web_search_options`` knob, so
    native web search must be expressed as an OpenAI tool definition instead.
    """
    tool_defs = function_tools_to_openai(tools, strict=strict, responses_api=responses_api) if tools else []
    if web_search_options is not None:
        tool_defs.append({"type": "web_search", **web_search_options})
    return tool_defs or None
