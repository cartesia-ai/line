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
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Type, Union, get_args, get_origin

from line.llm_agent.tools.utils import FunctionTool, ParameterInfo


def python_type_to_json_schema(type_annotation: Type) -> Dict[str, Any]:
    """
    Convert a Python type annotation to a JSON Schema type.

    Args:
        type_annotation: The Python type to convert.

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
        dict: {"type": "object"},
    }

    if type_annotation in type_map:
        return type_map[type_annotation]

    # Handle List[X]
    origin = get_origin(type_annotation)
    args = get_args(type_annotation)

    if origin is list:
        if args:
            return {"type": "array", "items": python_type_to_json_schema(args[0])}
        return {"type": "array"}

    # Handle Dict[K, V]
    if origin is dict:
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
            return python_type_to_json_schema(non_none_args[0])
        # For true Union types, use anyOf
        return {"anyOf": [python_type_to_json_schema(a) for a in non_none_args]}

    # Handle Enum types
    if isinstance(type_annotation, type) and issubclass(type_annotation, Enum):
        return {"type": "string", "enum": [e.value for e in type_annotation]}

    # Default to string for unknown types
    return {"type": "string"}


def build_parameters_schema(parameters: Dict[str, ParameterInfo]) -> Dict[str, Any]:
    """
    Build a JSON Schema for function parameters.

    Args:
        parameters: Dictionary of parameter info.

    Returns:
        A JSON Schema object describing the parameters.
    """
    properties = {}
    required = []

    for name, param in parameters.items():
        prop = python_type_to_json_schema(param.type_annotation)

        if param.description:
            prop["description"] = param.description

        if param.enum:
            prop["enum"] = param.enum

        if param.default is not None and not param.required:
            prop["default"] = param.default

        properties[name] = prop

        if param.required:
            required.append(name)

    schema = {"type": "object", "properties": properties}

    if required:
        schema["required"] = required

    return schema


def function_tool_to_litellm(tool: FunctionTool, *, strict: bool = True) -> Dict[str, Any]:
    """
    Convert a FunctionTool to LiteLLM (OpenAI Chat Completions) tool format.

    Args:
        tool: The FunctionTool to convert.
        strict: Whether to enable strict mode (default True).

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
        #     }
        # }
        ```
    """
    params_schema = build_parameters_schema(tool.parameters)

    if strict:
        params_schema["additionalProperties"] = False

    result: Dict[str, Any] = {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": params_schema,
        },
    }
    if strict:
        result["function"]["strict"] = True
    return result


def tools_to_litellm(tools: List[FunctionTool], *, strict: bool = True) -> List[Dict[str, Any]]:
    """
    Convert a list of FunctionTools to LiteLLM format.

    Args:
        tools: List of FunctionTool instances.
        strict: Whether to enable strict mode for function tools.

    Returns:
        List of tool definitions.
    """
    return [function_tool_to_litellm(t, strict=strict) for t in tools]
