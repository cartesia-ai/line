"""
Tool execution context for LLM agents.

This module provides the ToolContext class that is passed to tool functions
when they are executed, giving them access to conversation state and utilities.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from line.events import EventInstance


@dataclass
class ToolContext:
    """
    Context passed to tool functions during execution.

    This provides tools with access to conversation history, metadata,
    and utilities for common operations.

    Attributes:
        conversation_history: List of conversation events.
        metadata: Arbitrary metadata passed to the tool.
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

            # Access metadata
            user_id = ctx.metadata.get("user_id")

            return "result"
        ```
    """

    conversation_history: List[EventInstance] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None

    def get_last_user_message(self) -> Optional[str]:
        """
        Get the last user message from conversation history.

        Returns:
            The content of the last user message, or None if not found.
        """
        from line.events import UserTranscriptionReceived

        for event in reversed(self.conversation_history):
            if isinstance(event, UserTranscriptionReceived):
                return event.content
        return None

    def get_last_agent_message(self) -> Optional[str]:
        """
        Get the last agent message from conversation history.

        Returns:
            The content of the last agent message, or None if not found.
        """
        from line.events import AgentResponse

        for event in reversed(self.conversation_history):
            if isinstance(event, AgentResponse):
                return event.content
        return None


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
    events: List[EventInstance] = field(default_factory=list)
    handoff_target: Any = None

    @property
    def success(self) -> bool:
        """Check if the tool executed successfully."""
        return self.error is None
