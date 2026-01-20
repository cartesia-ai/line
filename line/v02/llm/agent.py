"""
Agent types and events for the LLM module.

Re-exports agent and event types from line.v02 for use in the LLM wrapper.
"""

from dataclasses import dataclass
from typing import Any, AsyncIterable, Awaitable, Protocol, Union

# Re-export agent types from v02
from line.v02.agent import (
    Agent,
    AgentCallable,
    AgentClass,
    AgentSpec,
    EventFilter,
    TurnEnv,
)

# Re-export all event types from v02
from line.v02.events import (
    # Input events with history
    AgentDTMFSent,
    # Output events
    AgentEndCall,
    AgentHandedOff,
    AgentSendDTMF,
    AgentSendText,
    AgentTextSent,
    AgentToolCalled,
    AgentToolCalledInput,
    AgentToolReturned,
    AgentToolReturnedInput,
    AgentTransferCall,
    AgentTurnEnded,
    AgentTurnStarted,
    CallEnded,
    CallStarted,
    InputEvent,
    LogMessage,
    LogMetric,
    OutputEvent,
    # Specific events
    SpecificAgentDTMFSent,
    SpecificAgentTextSent,
    SpecificAgentToolCalled,
    SpecificAgentToolReturned,
    SpecificAgentTurnEnded,
    SpecificAgentTurnStarted,
    SpecificCallEnded,
    SpecificCallStarted,
    SpecificInputEvent,
    SpecificUserDtmfSent,
    SpecificUserTextSent,
    SpecificUserTurnEnded,
    SpecificUserTurnStarted,
    UserDtmfSent,
    UserTextSent,
    UserTurnEnded,
    UserTurnStarted,
)


@dataclass
class ToolEnv:
    """Context passed to tool functions."""

    turn_env: TurnEnv


# Tool function protocols
# These define the expected signatures for each tool type


class LoopbackToolFn(Protocol):
    """Loopback tool: result is sent back to the LLM for continued generation.

    Signature: (ctx: ToolEnv, **kwargs) -> AsyncIterable[Any] | Awaitable[Any] | Any
    """

    def __call__(self, ctx: ToolEnv, /, **kwargs: Any) -> Union[AsyncIterable[Any], Awaitable[Any], Any]: ...


class PassthroughToolFn(Protocol):
    """Passthrough tool: response bypasses the LLM and goes directly to the user.

    Signature: (ctx: ToolEnv, **kwargs) ->
        AsyncIterable[OutputEvent] | Awaitable[OutputEvent] | OutputEvent
    """

    def __call__(
        self, ctx: ToolEnv, /, **kwargs: Any
    ) -> Union[AsyncIterable[OutputEvent], Awaitable[OutputEvent], OutputEvent]: ...


class HandoffToolFn(Protocol):
    """Handoff tool: transfers control to another agent.

    Signature: (ctx: ToolEnv, event: InputEvent, **kwargs) ->
        AsyncIterable[OutputEvent] | Awaitable[OutputEvent] | OutputEvent

    The event parameter receives AgentHandedOff on initial handoff,
    then subsequent InputEvents for continued processing.
    """

    def __call__(
        self, ctx: ToolEnv, /, event: InputEvent, **kwargs: Any
    ) -> Union[AsyncIterable[OutputEvent], Awaitable[OutputEvent], OutputEvent]: ...


__all__ = [
    # Agent types
    "Agent",
    "AgentCallable",
    "AgentClass",
    "AgentSpec",
    "EventFilter",
    "TurnEnv",
    # Tool types
    "ToolEnv",
    "LoopbackToolFn",
    "PassthroughToolFn",
    "HandoffToolFn",
    # Output events
    "AgentEndCall",
    "AgentSendDTMF",
    "AgentSendText",
    "AgentToolCalled",
    "AgentToolReturned",
    "AgentTransferCall",
    "LogMessage",
    "LogMetric",
    "OutputEvent",
    # LLM-specific events
    "AgentHandedOff",
    # Input events with history
    "AgentDTMFSent",
    "AgentTextSent",
    "AgentToolCalledInput",
    "AgentToolReturnedInput",
    "AgentTurnEnded",
    "AgentTurnStarted",
    "CallEnded",
    "CallStarted",
    "InputEvent",
    "UserDtmfSent",
    "UserTextSent",
    "UserTurnEnded",
    "UserTurnStarted",
    # Specific events
    "SpecificAgentDTMFSent",
    "SpecificAgentTextSent",
    "SpecificAgentToolCalled",
    "SpecificAgentToolReturned",
    "SpecificAgentTurnEnded",
    "SpecificAgentTurnStarted",
    "SpecificCallEnded",
    "SpecificCallStarted",
    "SpecificInputEvent",
    "SpecificUserDtmfSent",
    "SpecificUserTextSent",
    "SpecificUserTurnEnded",
    "SpecificUserTurnStarted",
]
