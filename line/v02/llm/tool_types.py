"""
Tool type decorators: @loopback_tool, @passthrough_tool, @handoff_tool.

See README.md for examples.
"""

from typing import Callable, Optional

from line.v02.llm.function_tool import construct_function_tool, FunctionTool, ToolType


def loopback_tool(
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable[[Callable], FunctionTool]:
    """
    Decorator for loopback tools. Result is sent back to the LLM.

    Use for information retrieval, calculations, API queries.
    Tool returns a value that the LLM incorporates into its response.
    """
    def decorator(func: Callable) -> FunctionTool:
        return construct_function_tool(func, name=name, description=description, tool_type=ToolType.LOOPBACK)
    return decorator


def passthrough_tool(
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable[[Callable], FunctionTool]:
    """
    Decorator for passthrough tools. Response bypasses the LLM.

    Use for deterministic actions like EndCall, TransferCall.
    Tool is an async generator that yields OutputEvent objects.
    """
    def decorator(func: Callable) -> FunctionTool:
        return construct_function_tool(func, name=name, description=description, tool_type=ToolType.PASSTHROUGH)
    return decorator


def handoff_tool(
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable[[Callable], FunctionTool]:
    """
    Decorator for handoff tools. Transfers control to another agent.

    Use for multi-agent workflows and department transfers.
    Tool is an async generator that yields events, then yields the target agent.
    """
    def decorator(func: Callable) -> FunctionTool:
        return construct_function_tool(func, name=name, description=description, tool_type=ToolType.HANDOFF)
    return decorator
