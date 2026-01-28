"""
Tool type decorators: @loopback_tool, @passthrough_tool, @handoff_tool.

Usage:
    @loopback_tool
    async def my_tool(ctx: ToolEnv, param: Annotated[str, "description"]):
        '''Tool description from docstring.'''
        ...

Works on both standalone functions and class methods.
"""

from functools import partial
from typing import Any, Callable

from line.v02.llm.tool_utils import FunctionTool, ToolType, construct_function_tool


class _ToolDescriptor(FunctionTool):
    """FunctionTool subclass that supports binding to class instances.

    Implements the descriptor protocol so that when a decorated method is
    accessed on an instance, it returns a FunctionTool with func bound to self.
    """

    def __get__(self, instance: Any, owner: type) -> FunctionTool:
        """Descriptor protocol: bind the tool's func to the instance."""
        if instance is None:
            return self
        # Return a plain FunctionTool with func bound to the instance
        return FunctionTool(
            name=self.name,
            description=self.description,
            func=partial(self.func, instance),
            parameters=self.parameters,
            tool_type=self.tool_type,
        )


def _construct_tool_descriptor(func: Callable, tool_type: ToolType) -> _ToolDescriptor:
    """Construct a _ToolDescriptor from a function."""
    base = construct_function_tool(
        func,
        name=func.__name__,
        description=(func.__doc__ or "").strip(),
        tool_type=tool_type,
    )
    return _ToolDescriptor(
        name=base.name,
        description=base.description,
        func=base.func,
        parameters=base.parameters,
        tool_type=base.tool_type,
    )


def loopback_tool(func: Callable) -> FunctionTool:
    """
    Decorator for loopback tools. Result is sent back to the LLM.

    Signature: (ctx: ToolEnv, **args) -> AsyncIterable[Any] | Awaitable[Any] | Any

    Use for information retrieval, calculations, API queries.
    Tool returns a value that the LLM incorporates into its response.
    """
    return _construct_tool_descriptor(func, ToolType.LOOPBACK)


def passthrough_tool(func: Callable) -> FunctionTool:
    """
    Decorator for passthrough tools. Response bypasses the LLM.

    Signature: (ctx: ToolEnv, **args) -> AsyncIterable[OutputEvent] | Awaitable[OutputEvent] | OutputEvent

    Use for deterministic actions like EndCall, TransferCall.
    Tool yields OutputEvent objects directly to the caller.
    """
    return _construct_tool_descriptor(func, ToolType.PASSTHROUGH)


def handoff_tool(func: Callable) -> FunctionTool:
    """
    Decorator for handoff tools. Transfers control to another process.

    Signature: (ctx: ToolEnv, **args) -> AsyncIterable[OutputEvent] | Awaitable[OutputEvent] | OutputEvent

    Use for multi-agent workflows or custom handlers.
    Tool yields OutputEvent objects and optionally yields the handoff target (AgentCallable).
    """
    return _construct_tool_descriptor(func, ToolType.HANDOFF)
