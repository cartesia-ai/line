"""
Base Agent class and types for the Line SDK.

This module defines the Agent protocol and related types for the conversational loop.
The Cartesia Agent Harness calls agents with InputEvents and consumes their OutputEvents.

An Agent can be either:
1. A class that implements `process(ctx, input) -> AsyncIterable[OutputEvent]`
2. A function with signature `(ctx, input) -> AsyncIterable[OutputEvent]`

Example:
    ```python
    # Class-based agent
    class MyAgent:
        async def process(self, ctx: TurnContext, event: InputEvent) -> AsyncIterable[OutputEvent]:
            if isinstance(event, UserTranscript):
                yield AgentOutput(text="Hello!")

    # Function-based agent
    async def my_agent(ctx: TurnContext, event: InputEvent) -> AsyncIterable[OutputEvent]:
        if isinstance(event, UserTranscript):
            yield AgentOutput(text="Hello!")

    # In get_agent
    def get_agent(ctx, call_request):
        return MyAgent()  # or my_agent
    ```
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterable,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)


# =============================================================================
# Input Events (from Cartesia server via Harness)
# =============================================================================


@dataclass
class InputEvent:
    """Base class for all input events from the harness."""

    pass


@dataclass
class CallStarted(InputEvent):
    """The call has started."""

    call_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CallEnded(InputEvent):
    """The call has ended."""

    reason: str = ""


@dataclass
class UserTurnStarted(InputEvent):
    """The user started speaking (interruption event)."""

    pass


@dataclass
class UserTurnEnded(InputEvent):
    """The user finished speaking."""

    pass


@dataclass
class UserTranscript(InputEvent):
    """Transcribed user speech."""

    text: str = ""
    is_final: bool = True


@dataclass
class UserDTMF(InputEvent):
    """DTMF input from user."""

    digit: str = ""


# =============================================================================
# Output Events (from Agent to Harness)
# =============================================================================


@dataclass
class OutputEvent:
    """Base class for all output events from agents."""

    pass


@dataclass
class AgentOutput(OutputEvent):
    """Text output to be spoken by the agent."""

    text: str = ""


@dataclass
class AgentEndTurn(OutputEvent):
    """Signal that the agent has finished its turn."""

    pass


@dataclass
class AgentEndCall(OutputEvent):
    """End the call."""

    reason: str = ""


@dataclass
class AgentTransferCall(OutputEvent):
    """Transfer the call to another number."""

    target: str = ""
    timeout_seconds: int = 30


@dataclass
class AgentSendDTMF(OutputEvent):
    """Send DTMF tones."""

    digits: str = ""


@dataclass
class AgentHandoff(OutputEvent):
    """Hand off to another agent."""

    target_agent: Any = None
    reason: str = ""


@dataclass
class ToolCallOutput(OutputEvent):
    """A tool was called (for observability)."""

    tool_name: str = ""
    tool_args: Dict[str, Any] = field(default_factory=dict)
    tool_call_id: str = ""


@dataclass
class ToolResultOutput(OutputEvent):
    """A tool returned a result (for observability)."""

    tool_name: str = ""
    tool_call_id: str = ""
    result: Any = None
    error: Optional[str] = None


# =============================================================================
# Turn Context
# =============================================================================


@dataclass
class TurnContext:
    """
    Context provided to agents for each turn.

    Contains conversation state, metadata, and utilities for the current turn.

    Attributes:
        call_id: Unique identifier for this call.
        turn_id: Unique identifier for this turn.
        conversation_history: List of previous events in this conversation.
        metadata: Arbitrary metadata for this call/turn.
    """

    call_id: str = ""
    turn_id: str = ""
    conversation_history: List[Union[InputEvent, OutputEvent]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_last_user_text(self) -> Optional[str]:
        """Get the last user transcript text."""
        for event in reversed(self.conversation_history):
            if isinstance(event, UserTranscript):
                return event.text
        return None

    def get_last_agent_text(self) -> Optional[str]:
        """Get the last agent output text."""
        for event in reversed(self.conversation_history):
            if isinstance(event, AgentOutput):
                return event.text
        return None


# =============================================================================
# Agent Protocol
# =============================================================================


@runtime_checkable
class AgentClass(Protocol):
    """Protocol for class-based agents."""

    def process(
        self, ctx: TurnContext, event: InputEvent
    ) -> AsyncIterable[OutputEvent]:
        """
        Process an input event and yield output events.

        Args:
            ctx: The turn context with conversation state.
            event: The input event to process.

        Yields:
            Output events (AgentOutput, AgentEndCall, etc.)
        """
        ...


# Type alias for function-based agents
AgentFunction = Callable[[TurnContext, InputEvent], AsyncIterable[OutputEvent]]

# An Agent is either a class with process() or a function
Agent = Union[AgentClass, AgentFunction]


# =============================================================================
# Turn Predicates
# =============================================================================

# Predicate types for customizing the loop
TurnStartPredicate = Union[
    List[type],  # List of event types that start a turn
    Callable[[InputEvent], bool],  # Function that returns True for turn start events
]

TurnEndPredicate = Union[
    List[type],  # List of event types that end/interrupt a turn
    Callable[[InputEvent], bool],  # Function that returns True for turn end events
]

# Default turn start events
DEFAULT_TURN_START_EVENTS = [CallStarted, UserTurnEnded, CallEnded]

# Default turn end/interruption events
DEFAULT_TURN_END_EVENTS = [UserTurnStarted]


def make_predicate(spec: Union[List[type], Callable[[InputEvent], bool]]) -> Callable[[InputEvent], bool]:
    """
    Convert a predicate specification to a callable.

    Args:
        spec: Either a list of event types or a predicate function.

    Returns:
        A predicate function.
    """
    if callable(spec):
        return spec
    return lambda ev: isinstance(ev, tuple(spec))
