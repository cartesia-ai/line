"""
Schema converter utilities for converting function tools to provider-specific formats.

This module provides functions to convert FunctionTool instances to the tool/function
calling formats expected by different LLM providers:

- OpenAI (Chat Completions API and Responses API)
- Anthropic (Claude API)
- Google (Gemini API)

Example:
    ```python
    from typing import Annotated
    from line.v02.llm import loopback_tool
    from line.v02.llm.schema_converter import (
        function_tool_to_openai,
        function_tool_to_anthropic,
        function_tool_to_gemini,
    )

    @loopback_tool()
    async def my_tool(ctx, param: Annotated[str, "Parameter description"]):
        '''Tool description'''
        ...

    openai_tool = function_tool_to_openai(my_tool)
    anthropic_tool = function_tool_to_anthropic(my_tool)
    gemini_tool = function_tool_to_gemini(my_tool)
    ```
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Type, Union, get_args, get_origin

from line.v02.llm.function_tool import FunctionTool, ParameterInfo


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


def function_tool_to_openai(
    tool: FunctionTool, *, strict: bool = True, responses_api: bool = False
) -> Dict[str, Any]:
    """
    Convert a FunctionTool to OpenAI tool format.

    Args:
        tool: The FunctionTool to convert.
        strict: Whether to enable strict mode (default True).
        responses_api: If True, use the Responses API format; otherwise Chat Completions.

    Returns:
        OpenAI tool definition dictionary.

    Example:
        ```python
        @loopback_tool()
        async def get_weather(ctx, city: Annotated[str, "City name"]):
            '''Get the weather'''
            ...

        # Chat Completions API format
        openai_tool = function_tool_to_openai(get_weather)
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

        # Responses API format
        openai_tool = function_tool_to_openai(get_weather, responses_api=True)
        # Returns:
        # {
        #     "type": "function",
        #     "name": "get_weather",
        #     "description": "Get the weather",
        #     "parameters": {...},
        #     "strict": True
        # }
        ```
    """
    params_schema = build_parameters_schema(tool.parameters)

    if strict:
        params_schema["additionalProperties"] = False

    if responses_api:
        # Responses API format (flat structure)
        result: Dict[str, Any] = {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": params_schema,
        }
        if strict:
            result["strict"] = True
        return result
    else:
        # Chat Completions API format (nested under "function")
        result = {
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


def function_tool_to_anthropic(tool: FunctionTool) -> Dict[str, Any]:
    """
    Convert a FunctionTool to Anthropic Claude tool format.

    Args:
        tool: The FunctionTool to convert.

    Returns:
        Anthropic tool definition dictionary.

    Example:
        ```python
        @loopback_tool()
        async def get_weather(ctx, city: Annotated[str, "City name"]):
            '''Get the weather'''
            ...

        anthropic_tool = function_tool_to_anthropic(get_weather)
        # Returns:
        # {
        #     "name": "get_weather",
        #     "description": "Get the weather",
        #     "input_schema": {
        #         "type": "object",
        #         "properties": {...},
        #         "required": [...]
        #     }
        # }
        ```
    """
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": build_parameters_schema(tool.parameters),
    }


def function_tool_to_gemini(tool: FunctionTool) -> Any:
    """
    Convert a FunctionTool to Google Gemini tool format.

    Args:
        tool: The FunctionTool to convert.

    Returns:
        Gemini Tool object.

    Example:
        ```python
        @loopback_tool()
        async def get_weather(ctx, city: Annotated[str, "City name"]):
            '''Get the weather'''
            ...

        gemini_tool = function_tool_to_gemini(get_weather)
        # Returns a gemini_types.Tool object
        ```
    """
    try:
        from google.genai import types as gemini_types
    except ImportError as e:
        raise ImportError(
            "google-genai is required for Gemini integration. Install with: pip install google-genai"
        ) from e

    params_schema = build_parameters_schema(tool.parameters)

    return gemini_types.Tool(
        function_declarations=[
            gemini_types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=params_schema,
            )
        ]
    )


def function_tools_to_openai(
    tools: List[FunctionTool], *, strict: bool = True, responses_api: bool = False
) -> List[Dict[str, Any]]:
    """
    Convert multiple FunctionTools to OpenAI format.

    Args:
        tools: List of FunctionTools to convert.
        strict: Whether to enable strict mode.
        responses_api: If True, use the Responses API format.

    Returns:
        List of OpenAI tool definitions.
    """
    return [function_tool_to_openai(t, strict=strict, responses_api=responses_api) for t in tools]


def function_tools_to_anthropic(tools: List[FunctionTool]) -> List[Dict[str, Any]]:
    """
    Convert multiple FunctionTools to Anthropic format.

    Args:
        tools: List of FunctionTools to convert.

    Returns:
        List of Anthropic tool definitions.
    """
    return [function_tool_to_anthropic(t) for t in tools]


def function_tools_to_gemini(tools: List[FunctionTool]) -> List[Any]:
    """
    Convert multiple FunctionTools to Gemini format.

    Note: Gemini prefers all function declarations in a single Tool object.
    This function returns individual Tool objects; use `merge_gemini_tools`
    to combine them if needed.

    Args:
        tools: List of FunctionTools to convert.

    Returns:
        List of Gemini Tool objects.
    """
    return [function_tool_to_gemini(t) for t in tools]


def merge_gemini_tools(tools: List[Any]) -> Any:
    """
    Merge multiple Gemini Tool objects into a single Tool.

    Gemini works best when all function declarations are in a single Tool object.

    Args:
        tools: List of Gemini Tool objects.

    Returns:
        A single Gemini Tool with all function declarations.
    """
    try:
        from google.genai import types as gemini_types
    except ImportError as e:
        raise ImportError(
            "google-genai is required for Gemini integration. Install with: pip install google-genai"
        ) from e

    all_declarations = []
    for tool in tools:
        if hasattr(tool, "function_declarations"):
            all_declarations.extend(tool.function_declarations)

    return gemini_types.Tool(function_declarations=all_declarations)
