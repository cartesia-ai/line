"""
Agent types and events for the LLM module.

This module re-exports agent and event types from line.v02 for use in the LLM wrapper,
plus defines LLM-specific extensions for tool call tracking and handoff.
"""

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

# Re-export agent types from v02 (relative import since we're in v02)
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
    # Input events with history (harness -> agent)
    AgentDTMFSent,
    # Output events (agent -> harness)
    AgentEndCall,
    AgentSendDTMF,
    AgentSendText,
    AgentTextSent,
    AgentToolCalled,
    AgentToolReturned,
    AgentTransferCall,
    AgentTurnEnded,
    AgentTurnStarted,
    CallEnded,
    CallStarted,
    InputEvent,
    LogMessage,
    LogMetric,
    OutputEvent,
    # Specific events (without history, used in history lists)
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

# =============================================================================
# LLM-specific event extensions
# These events extend v02 types with additional fields needed for LLM tool tracking.
# =============================================================================


class ToolCallEvent(BaseModel):
    """
    Extended tool call event with tool_call_id for LLM result matching.

    This extends AgentToolCalled with the tool_call_id field that LLMs use
    to match tool results back to their corresponding calls.
    """

    type: Literal["tool_call"] = "tool_call"
    tool_name: str
    tool_args: Dict[str, Any] = Field(default_factory=dict)
    tool_call_id: Optional[str] = None


class ToolResultEvent(BaseModel):
    """
    Extended tool result event with tool_call_id and error field.

    This extends AgentToolReturned with tool_call_id for matching and
    an error field for capturing tool execution failures.
    """

    type: Literal["tool_result"] = "tool_result"
    tool_name: str
    tool_call_id: Optional[str] = None
    result: Any = None
    error: Optional[str] = None


class AgentHandoff(BaseModel):
    """
    Event emitted when control is transferred to another agent.

    This is an LLM-specific event for the handoff tool paradigm where
    one agent transfers control to another.
    """

    type: Literal["agent_handoff"] = "agent_handoff"
    target_agent: str
    reason: Optional[str] = None


__all__ = [
    # Agent types
    "Agent",
    "AgentCallable",
    "AgentClass",
    "AgentSpec",
    "EventFilter",
    "TurnEnv",
    # Output events (v02)
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
    "ToolCallEvent",
    "ToolResultEvent",
    "AgentHandoff",
    # Input events with history
    "AgentDTMFSent",
    "AgentTextSent",
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
