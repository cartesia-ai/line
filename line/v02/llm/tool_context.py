"""
Tool execution context for LLM agents.

This module provides the ToolContext class that is passed to tool functions
when they are executed, giving them access to conversation state and utilities.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class ToolContext:
    """
    Context passed to tool functions during execution.

    Attributes:
        conversation_history: List of conversation events (SpecificInputEvent types from v02).
        tool_call_id: The ID of the current tool call.
        tool_name: The name of the tool being called.

    Example:
        ```python
        @function_tool
        async def my_tool(
            ctx: ToolContext,
            param: Annotated[str, Field(description="...")]
        ):
            # Access conversation history
            for event in ctx.conversation_history:
                print(event)
            return "result"
        ```
    """

    conversation_history: List[Any] = field(default_factory=list)  # List of SpecificInputEvent
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None


@dataclass
class ToolResult:
    """
    Result from executing a tool.

    Attributes:
        tool_call_id: The ID of the tool call.
        tool_name: The name of the tool.
        result: The result value from the tool.
        error: Error message if the tool failed.
        events: Events yielded by passthrough tools.
        handoff_target: The handler to hand off to (for handoff tools).
    """

    tool_call_id: str
    tool_name: str
    result: Any = None
    error: Optional[str] = None
    events: List[Any] = field(default_factory=list)  # OutputEvent types from passthrough tools
    handoff_target: Any = None

    @property
    def success(self) -> bool:
        """Check if the tool executed successfully."""
        return self.error is None
