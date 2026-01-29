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
from typing import Any, Callable, Union

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
            is_background=self.is_background,
        )


def _construct_tool_descriptor(
    func: Callable, tool_type: ToolType, is_background: bool = False
) -> _ToolDescriptor:
    """Construct a _ToolDescriptor from a function."""
    base = construct_function_tool(
        func,
        name=func.__name__,
        description=(func.__doc__ or "").strip(),
        tool_type=tool_type,
        is_background=is_background,
    )
    return _ToolDescriptor(
        name=base.name,
        description=base.description,
        func=base.func,
        parameters=base.parameters,
        tool_type=base.tool_type,
        is_background=base.is_background,
    )


def loopback_tool(
    func: Callable = None, *, is_background: bool = False
) -> Union[FunctionTool, Callable[[Callable], FunctionTool]]:
    """
    Decorator for loopback tools. Result is sent back to the LLM to trigger a new completion.

    Signature: (ctx: ToolEnv, **args) -> AsyncIterable[Any] | Awaitable[Any] | Any

    Can be used with or without arguments:
        @loopback_tool
        def my_tool(ctx: ToolEnv, ...): ...

        @loopback_tool(is_background=True)
        def my_background_tool(ctx: ToolEnv, ...): ...

    If is_background=True:
    1) the tool runs asynchronously without blocking the LLM/other tool calls,
        triggering a new completion on each yielded value
    2) the tool is not cancelled on interruption, any yielded values are
        incorporated into the conversation history for future completions.

    Example:
    @loopback_tool(is_background=True)
    async def check_bank_balance(ctx, order_id):
        yield "bank balance look up pending"
        balance = await take_long_time_to_get_balance(order_id)
        yield balance

    1) The user asks "What's my bank balance?"
    2) The LLM calls check_bank_balance(order_id=12345)
    3) The tool yields "bank balance look up pending" immediately, triggering a new completion.
    4) The LLM responds with "Please wait while I check your balance."
    5) Option 1: The user responds with "Thanks, I can wait".
        a) During the completion in response to 5, the tool completes the balance lookup.
        b) The tool yields the balance, triggering another completion.
        c) The LLM responds with "Your balance is $1,234.56
    6) Option 2: The user never responds.
        a) The tool completes the balance lookup in the background.
        b) The tool yields the balance, triggering another completion.
        c) The LLM responds with "Thank you for waiting. Your balance is $1,234.56"
    NOTE: this means the tool response becomes available to the LLM to power completions
    whether or not the user responds.

    """

    def decorator(f: Callable) -> FunctionTool:
        return _construct_tool_descriptor(f, ToolType.LOOPBACK, is_background=is_background)

    if func is not None:
        # Called without arguments: @loopback_tool
        return decorator(func)
    # Called with arguments: @loopback_tool(is_background=True)
    return decorator


def passthrough_tool(func: Callable) -> FunctionTool:
    """
    Decorator for passthrough tools. Response bypasses the LLM, and is sent directly downstream

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
