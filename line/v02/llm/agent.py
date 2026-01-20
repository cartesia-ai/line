"""
Agent types and events for the LLM module.

Re-exports agent and event types from line.v02 for use in the LLM wrapper.
"""

from dataclasses import dataclass
from typing import Any, AsyncIterable, Awaitable, Callable, Union

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
class ToolContext:
    """Context passed to tool functions."""

    turn_env: TurnEnv


# Tool function type aliases
# Loopback: (ToolContext, **Args) => AsyncIterable[Any] | Awaitable[Any] | Any
LoopbackToolFn = Callable[..., Union[AsyncIterable[Any], Awaitable[Any], Any]]

# Passthrough: (ToolContext, **Args) => AsyncIterable[OutputEvent] | Awaitable[OutputEvent] | OutputEvent
PassthroughToolFn = Callable[..., Union[AsyncIterable[OutputEvent], Awaitable[OutputEvent], OutputEvent]]

# Handoff: (ToolContext, **Args) => AsyncIterable[OutputEvent] | Awaitable[OutputEvent] | OutputEvent
HandoffToolFn = Callable[..., Union[AsyncIterable[OutputEvent], Awaitable[OutputEvent], OutputEvent]]


__all__ = [
    # Agent types
    "Agent",
    "AgentCallable",
    "AgentClass",
    "AgentSpec",
    "EventFilter",
    "TurnEnv",
    # Tool types
    "ToolContext",
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
