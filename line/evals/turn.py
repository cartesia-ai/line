"""
Turn-based conversation representation for evaluation.

This module provides Turn classes that represent conversation turns with automatic
conversion to/from Event instances for use with ReasoningNode testing.
"""

import json
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from line.evals.similarity_utils import is_similar_dict, is_similar_text
from line.events import (
    AgentResponse,
    DTMFOutputEvent,
    EndCall,
    EventInstance,
    ToolResult,
    TransferCall,
    UserTranscriptionReceived,
)
from line.events import (
    ToolCall as EventToolCall,
)


class ToolCall(BaseModel):
    """Tool call representation within a Turn."""

    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    result: Any = None


class Turn(BaseModel):
    """Base class for conversation turns with event conversion capabilities."""

    role: Literal["user", "assistant"]
    text: Union[List[str], str] = ""
    tool_calls: List[ToolCall] = Field(default_factory=list)
    telephony_events: list[Union[DTMFOutputEvent, TransferCall, EndCall]] = Field(default_factory=list)

    @property
    def is_user(self) -> bool:
        """Check if this is a user turn."""
        return self.role == "user"

    @property
    def is_agent(self) -> bool:
        """Check if this is an agent turn."""
        return self.role == "assistant"

    def to_events(self) -> List[EventInstance]:
        """Convert this turn to a list of Event instances."""
        events = []

        if self.role == "user":
            if isinstance(self.text, str):
                events.append(UserTranscriptionReceived(content=self.text))
                return events

            # Otherwise, it must be a list
            if len(self.text) != 1:
                raise RuntimeError("Must include exactly one text element for user turn. {len(self.text)=}")
            if self.text:
                # Join all text elements with a space for user transcription
                events.append(UserTranscriptionReceived(content=self.text[0]))
        elif self.role == "assistant":
            # Add tool calls first
            for tool_call in self.tool_calls:
                events.append(EventToolCall(tool_name=tool_call.name, tool_args=tool_call.arguments))
                if tool_call.result is not None:
                    events.append(
                        ToolResult(
                            tool_name=tool_call.name,
                            tool_args=tool_call.arguments,
                            result=tool_call.result,
                        )
                    )

            # Add text response
            if self.text:
                if isinstance(self.text, str):
                    events.append(AgentResponse(content=self.text))
                elif isinstance(self.text, list):
                    events.append(AgentResponse(content=self.text[0]))
                else:
                    raise RuntimeError(f"Unexpected text type: {type(self.text)=}")

        return events

    @classmethod
    def from_events(cls, events: List[EventInstance]) -> "Turn":
        """Create a Turn from a list of Event instances."""
        text = ""
        tool_calls = []
        role = "assistant"  # Default to assistant

        # Track tool calls and their results
        tool_call_map = {}
        telephony_events = []

        for event in events:
            if isinstance(event, UserTranscriptionReceived):
                role = "user"
                text += event.content
            elif isinstance(event, AgentResponse):
                role = "assistant"
                text += event.content
            elif isinstance(event, EventToolCall):
                role = "assistant"
                tool_call_map[event.tool_name] = ToolCall(name=event.tool_name, arguments=event.tool_args)
            elif isinstance(event, ToolResult):
                role = "assistant"
                if event.tool_name in tool_call_map:
                    tool_call_map[event.tool_name].result = event.result
                else:
                    # Create tool call if we only have the result
                    tool_call_map[event.tool_name] = ToolCall(
                        name=event.tool_name,
                        arguments=event.tool_args,
                        result=event.result,
                    )
            elif (
                isinstance(event, DTMFOutputEvent)
                or isinstance(event, TransferCall)
                or isinstance(event, EndCall)
            ):
                role = "assistant"
                telephony_events.append(event)

        tool_calls = list(tool_call_map.values())
        text = text.strip()

        return cls(role=role, text=text, tool_calls=tool_calls, telephony_events=telephony_events)

    def is_similar(self, other: "Turn") -> Optional[str]:
        """Check if this turn is similar to another turn.

        Returns:
            None if turns are similar, error description string if not
        """
        # Check role matches
        if self.role != other.role:
            return f"Role mismatch: expected '{other.role}', got '{self.role}'"

        # Check text similarity
        if self.text or other.text:
            results = is_similar_text(self.text, other.text)
            if results.is_success is False:
                return f"Text mismatch: {results.error}"

        # Check tool calls match
        if len(self.tool_calls) != len(other.tool_calls):
            return f"Tool call count mismatch: expected {len(other.tool_calls)}, got {len(self.tool_calls)}"

        # Sort tool calls by name for comparison
        self_tools = sorted(self.tool_calls, key=lambda x: x.name)
        other_tools = sorted(other.tool_calls, key=lambda x: x.name)

        for self_tool, other_tool in zip(self_tools, other_tools):
            if self_tool.name != other_tool.name:
                return f"Tool name mismatch: expected '{other_tool.name}', got '{self_tool.name}'"

            # Check arguments similarity
            if self_tool.arguments or other_tool.arguments:
                results = is_similar_dict(self_tool.arguments, other_tool.arguments)
                if results.is_success is False:
                    return f"Tool '{self_tool.name}' arguments mismatch: {results.error}"

            # Check result similarity
            if self_tool.result != other_tool.result:
                return (
                    f"Tool '{self_tool.name}' result mismatch: "
                    f"expected {other_tool.result}, got {self_tool.result}"
                )

        if self.telephony_events != other.telephony_events:
            return f"telephony_events mismatch: expected {other.telephony_events} to match {self.telephony_events}"  # noqa: E501

        return None


class UserTurn(Turn):
    """User conversation turn."""

    role: Literal["user"] = "user"


class AgentTurn(Turn):
    """Agent conversation turn."""

    role: Literal["assistant"] = "assistant"


def make_turn(data: Dict[str, Any]) -> Union[UserTurn, AgentTurn]:
    """Create a UserTurn or AgentTurn from dictionary data.

    Args:
        data: Dictionary containing turn data with 'role' field and other turn properties

    Returns:
        UserTurn or AgentTurn instance based on the role

    Raises:
        ValueError: If role is not 'user' or 'assistant'
    """
    role = data.get("role")

    if role == "user":
        return UserTurn(**data)
    elif role == "assistant":
        return AgentTurn(**data)
    else:
        raise ValueError(f"Invalid role '{role}'. Must be 'user' or 'assistant'")


def load_conversation_json(file_path: str) -> List[Union[UserTurn, AgentTurn]]:
    """Load a conversation from a JSON file.

    Args:
        file_path: Path to JSON file containing conversation data

    Returns:
        List of Turn instances (UserTurn or AgentTurn)

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
        ValueError: If any turn has an invalid role
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    return [make_turn(turn_data) for turn_data in data]
