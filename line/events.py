"""
Typed event definitions for the agent bus system.

Each event inherits from EventMeta which provides automatic node identification
and instance tracking for distributed agent communication.
"""

import json
from typing import Any, Dict, Optional, Type, TypeVar, Union
import uuid

from pydantic import BaseModel, Field

T = TypeVar("T")
EventInstance = T

# Type[T] means EventType has to be a class (not an instance), and it allows us to refer to T in
# order to instantiate T later.
EventType = Type[T]

EventTypeOrAlias = Union[EventType, str]

__all__ = [
    "AgentResponse",
    "ToolResult",
    "ToolCall",
    "EndCall",
    "AgentGenerationComplete",
    "Authorize",
    "AgentError",
    "TransferCall",
    "AgentHandoff",
    "AgentStartedSpeaking",
    "AgentStoppedSpeaking",
    "UserStartedSpeaking",
    "UserStoppedSpeaking",
    "UserTranscriptionReceived",
    "AgentSpeechSent",
    "UserUnknownInputReceived",
    "LogMetric",
    "DTMFInputEvent",
    "DTMFOutputEvent",
    "DTMFStoppedEvent",
]


class AgentResponse(BaseModel):
    """Agent message to be sent to the user."""

    content: str
    chunk_type: str = "text"


class ToolResult(BaseModel):
    """Tool execution result
    - This will appear in the transcript in the Agent's current turn.

    Attributes:
    - tool_name: Name of the tool that was called.
    - tool_args: Arguments that were passed to the tool.
    - result: Result returned by the tool.
    - result_str: String representation of the result (computed).
    - error: Error message if the tool call failed (None if successful).
    - metadata: Additional metadata about the tool call.
    - tool_call_id: Reference to the ToolCall instance that triggered this result (if applicable).
    """

    tool_name: str = ""
    tool_args: dict = Field(default_factory=dict)
    result: Optional[object] = None
    error: Optional[str] = None
    metadata: Optional[Dict] = None
    tool_call_id: Optional[str] = None

    @property
    def result_str(self) -> Optional[str]:
        """String representation of the result, automatically computed from result."""
        if self.result is not None:
            try:
                return json.dumps(self.result)
            except Exception:
                return str(self.result)
        return None

    @property
    def success(self) -> bool:
        """Returns True if there was no error, False otherwise."""
        return self.error is None


class ToolCall(BaseModel):
    """Tool execution request
    - This will appear in the transcript in the Agent's current turn.

    Attributes:
    - tool_name: Name of the tool that was called
    - tool_args: Arguments that were passed to the tool
    - tool_call_id: Unique identifier for the tool call
    - raw_response: Raw response from the tool call
    """

    tool_name: str
    tool_args: Dict = Field(default_factory=dict)
    tool_call_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    raw_response: Dict = Field(default_factory=dict)


class EndCall(BaseModel):
    """End the call."""

    @property
    def content(self) -> str:
        """Returns string representation of the end call event."""
        return self.__repr__()


class AgentGenerationComplete(BaseModel):
    """Agent generation completion event."""


class Authorize(BaseModel):
    """Change the authorized agent."""

    agent: str


class AgentError(BaseModel):
    """Send error message to user."""

    error: str
    code: Optional[str] = None


class TransferCall(BaseModel):
    """Initiate transfer call to destination. When that happens:
    1. A transfer request will be sent
    2. The agent will wait for a timeout period (so the second leg has the opportunity to connect)
    3. Afterwards, the agent harness will begin shutdown

    For twilio clients:
    - If the second leg has not connected by the timeout period, #3 will terminate the call for all parties.
    - If the second leg connects, once #3 occurs, the agent will shut down and the transferred call continues.
    """

    target_phone_number: str
    timeout_s: Optional[int] = 30


class AgentHandoff(BaseModel):
    """Agent handoff event for transfer_to_* patterns."""

    target_agent: str
    reason: str = ""


class AgentStartedSpeaking(BaseModel):
    """Agent started speaking event."""


class AgentStoppedSpeaking(BaseModel):
    """Agent stopped speaking event."""


class UserStartedSpeaking(BaseModel):
    """User started speaking event."""


class UserStoppedSpeaking(BaseModel):
    """User stopped speaking event."""


class UserTranscriptionReceived(BaseModel):
    """User transcription received event."""

    content: str


class AgentSpeechSent(BaseModel):
    """Agent speech content sent event."""

    content: str


class UserUnknownInputReceived(BaseModel):
    """User unknown input received event."""

    input_data: str


class LogMetric(BaseModel):
    """Log metric event for tracking usage metrics."""

    name: str
    value: Any


class DTMFInputEvent(BaseModel):
    """DTMF event for tracking DTMF input."""

    button: str


class DTMFOutputEvent(BaseModel):
    """DTMF event for tracking DTMF input."""

    button: str


class DTMFStoppedEvent(BaseModel):
    """DTMF stopped event for tracking DTMF input."""


class _EventsRegistry:
    """A singleton registry of all events.

    Usage:
        >>> registry = EventsRegistry()
        >>> registry.register("system.eventA", SystemEventA)
        >>> registry.register("system.eventB", SystemEventB)
        >>> registry.get("system.eventA")
        <class 'system.eventA'>
        >>> registry.get("system.eventB")
        <class 'system.eventB'>
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.events = {}  # Dict[EventType, str]
        return cls._instance

    def register(self, alias: str, event_type: EventType):
        if event_type in self.events:
            raise ValueError(f"Event type {event_type} already registered with alias {alias}")
        if not isinstance(alias, str):
            raise TypeError(f"Alias {alias} is not a string")
        self.events[event_type] = alias

    def get(self, event_type: EventType) -> Optional[str]:
        return self.events.get(event_type, None)


EventsRegistry = _EventsRegistry()
