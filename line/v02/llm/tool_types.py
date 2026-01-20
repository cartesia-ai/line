"""
Tool type decorators: @loopback_tool, @passthrough_tool, @handoff_tool.

See README.md for examples.
"""

from typing import Callable, Optional

from line.v02.llm.function_tool import FunctionTool, ToolType, construct_function_tool


def loopback_tool(
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable[[Callable], FunctionTool]:
    """
    Decorator for loopback tools. Result is sent back to the LLM.

    Signature: (ctx: ToolEnv, **args) -> AsyncIterable[Any] | Awaitable[Any] | Any

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

    Signature: (ctx: ToolEnv, **args) -> AsyncIterable[OutputEvent] | Awaitable[OutputEvent] | OutputEvent

    Use for deterministic actions like EndCall, TransferCall.
    Tool yields OutputEvent objects directly to the caller.
    """

    def decorator(func: Callable) -> FunctionTool:
        return construct_function_tool(
            func, name=name, description=description, tool_type=ToolType.PASSTHROUGH
        )

    return decorator


def handoff_tool(
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable[[Callable], FunctionTool]:
    """
    Decorator for handoff tools. Transfers control to another process.

    Signature: (ctx: ToolEnv, **args) -> AsyncIterable[OutputEvent] | Awaitable[OutputEvent] | OutputEvent

    Use for multi-agent workflows or custom handlers.
    Tool yields OutputEvent objects and optionally yields the handoff target (AgentCallable).
    """

    def decorator(func: Callable) -> FunctionTool:
        return construct_function_tool(func, name=name, description=description, tool_type=ToolType.HANDOFF)

    return decorator
