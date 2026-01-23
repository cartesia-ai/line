"""
Tool type decorators: @loopback_tool, @passthrough_tool, @handoff_tool.

Usage:
    @loopback_tool
    async def my_tool(ctx: ToolEnv, param: Annotated[str, "description"]):
        '''Tool description from docstring.'''
        ...
"""

from typing import Callable

from line.v02.llm.tool_utils import FunctionTool, ToolType, construct_function_tool


def loopback_tool(func: Callable) -> FunctionTool:
    """
    Decorator for loopback tools. Result is sent back to the LLM.

    Signature: (ctx: ToolEnv, **args) -> AsyncIterable[Any] | Awaitable[Any] | Any

    Use for information retrieval, calculations, API queries.
    Tool returns a value that the LLM incorporates into its response.
    """
    return construct_function_tool(
        func,
        name=func.__name__,
        description=func.__doc__ or "",
        tool_type=ToolType.LOOPBACK,
    )


def passthrough_tool(func: Callable) -> FunctionTool:
    """
    Decorator for passthrough tools. Response bypasses the LLM.

    Signature: (ctx: ToolEnv, **args) -> AsyncIterable[OutputEvent] | Awaitable[OutputEvent] | OutputEvent

    Use for deterministic actions like EndCall, TransferCall.
    Tool yields OutputEvent objects directly to the caller.
    """
    return construct_function_tool(
        func,
        name=func.__name__,
        description=func.__doc__ or "",
        tool_type=ToolType.PASSTHROUGH,
    )


def handoff_tool(func: Callable) -> FunctionTool:
    """
    Decorator for handoff tools. Transfers control to another process.

    Signature: (ctx: ToolEnv, **args) -> AsyncIterable[OutputEvent] | Awaitable[OutputEvent] | OutputEvent

    Use for multi-agent workflows or custom handlers.
    Tool yields OutputEvent objects and optionally yields the handoff target (AgentCallable).
    """
    return construct_function_tool(
        func,
        name=func.__name__,
        description=func.__doc__ or "",
        tool_type=ToolType.HANDOFF,
    )
