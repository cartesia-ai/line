"""
Tool type decorators for LLM agents.

This module provides decorators that define how tool responses affect the flow
of events in the agent system:

- @loopback_tool (default): Response loops back to the LLM for continued generation
- @passthrough_tool: Response is emitted directly, bypassing the LLM
- @handoff_tool: Future events are routed to the tool (typically another agent)

Example:
    ```python
    from line.llm import loopback_tool, passthrough_tool, handoff_tool, Field
    from typing import Annotated

    # Default behavior - response goes back to LLM
    @loopback_tool
    async def get_weather(
        ctx: ToolContext,
        city: Annotated[str, Field(description="The city name")]
    ) -> str:
        '''Get the current weather for a city'''
        return f"72°F and sunny in {city}"

    # Response bypasses LLM, goes directly to user
    @passthrough_tool
    async def end_call(
        ctx: ToolContext,
        message: Annotated[str, Field(description="Goodbye message")]
    ):
        '''End the call with a message'''
        yield AgentResponse(content=message)
        yield EndCall()

    # Control is handed off to another agent/handler
    @handoff_tool
    async def transfer_to_sales(ctx: ToolContext):
        '''Transfer the conversation to the sales team'''
        return SalesAgent()
    ```
"""

from typing import Callable, Optional, Union

from line.llm.function_tool import FunctionTool, ToolType


def loopback_tool(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Union[FunctionTool, Callable[[Callable], FunctionTool]]:
    """
    Decorator for loopback tools.

    On calling the tool, the response is looped back to the calling LLM,
    which can choose to continue generating. This is the default behavior
    for tools.

    Use this for tools that:
    - Fetch information the LLM needs to formulate a response
    - Perform calculations the LLM should incorporate
    - Query APIs whose results should be summarized

    Example:
        ```python
        @loopback_tool
        async def get_temperature(
            ctx: ToolContext,
            city: Annotated[str, Field(description="The city name")]
        ) -> str:
            '''Get the current temperature in a city'''
            temp = await weather_api.get_temp(city)
            return f"{temp}°F"
        ```

    Args:
        func: The function to wrap.
        name: Override the tool name.
        description: Override the description.

    Returns:
        A FunctionTool with loopback behavior.
    """

    def decorator(f: Callable) -> FunctionTool:
        return FunctionTool(f, name=name, description=description, tool_type=ToolType.LOOPBACK)

    if func is not None:
        return decorator(func)

    return decorator


def passthrough_tool(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Union[FunctionTool, Callable[[Callable], FunctionTool]]:
    """
    Decorator for passthrough tools.

    On calling the tool, the response is emitted directly out to the user,
    NOT passed back to the LLM. Use this for:

    - Commands to the agent harness (EndCall, TransferCall, SendDtmf)
    - Deterministic responses that don't need LLM processing
    - Form-filling agents that choose the next question deterministically

    The tool should be an async generator that yields events.

    Example:
        ```python
        @passthrough_tool
        async def end_call(
            ctx: ToolContext,
            goodbye_message: Annotated[str, Field(description="A farewell message")]
        ):
            '''End the call with a goodbye message'''
            yield AgentResponse(content=goodbye_message)
            yield EndCall()
        ```

    Args:
        func: The function to wrap.
        name: Override the tool name.
        description: Override the description.

    Returns:
        A FunctionTool with passthrough behavior.
    """

    def decorator(f: Callable) -> FunctionTool:
        return FunctionTool(f, name=name, description=description, tool_type=ToolType.PASSTHROUGH)

    if func is not None:
        return decorator(func)

    return decorator


def handoff_tool(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Union[FunctionTool, Callable[[Callable], FunctionTool]]:
    """
    Decorator for handoff tools.

    Once the tool is called, all future events are routed to the tool's
    returned handler (typically another agent). The calling LLM no longer
    receives events after a handoff.

    Use this for:
    - Transferring to another specialized agent
    - Delegating to a sub-agent for a specific task
    - Multi-agent workflows

    The tool should return an agent or handler that will receive future events.

    Example:
        ```python
        @handoff_tool
        async def transfer_to_billing(
            ctx: ToolContext,
            reason: Annotated[str, Field(description="Why transferring")]
        ):
            '''Transfer the conversation to the billing department'''
            return BillingAgent(context=ctx.conversation_history)
        ```

    Args:
        func: The function to wrap.
        name: Override the tool name.
        description: Override the description.

    Returns:
        A FunctionTool with handoff behavior.
    """

    def decorator(f: Callable) -> FunctionTool:
        return FunctionTool(f, name=name, description=description, tool_type=ToolType.HANDOFF)

    if func is not None:
        return decorator(func)

    return decorator
