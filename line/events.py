"""Typed event definitions"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
import uuid

from pydantic import BaseModel, Field


def _generate_event_id() -> str:
    """Generate a stable UUID for an event."""
    return str(uuid.uuid4())


# -------------------------
# Output Events (agent -> harness)
# -------------------------


class AgentSendText(BaseModel):
    type: Literal["agent_send_text"] = "agent_send_text"
    text: str


class AgentToolCalled(BaseModel):
    type: Literal["agent_tool_called"] = "agent_tool_called"
    tool_call_id: str
    tool_name: str
    tool_args: Dict[str, Any] = Field(default_factory=dict)


class AgentToolReturned(BaseModel):
    type: Literal["agent_tool_returned"] = "agent_tool_returned"
    tool_call_id: str
    tool_name: str
    tool_args: Dict[str, Any] = Field(default_factory=dict)
    result: Any = None


class AgentEndCall(BaseModel):
    type: Literal["end_call"] = "end_call"


class AgentTransferCall(BaseModel):
    type: Literal["agent_transfer_call"] = "agent_transfer_call"
    target_phone_number: str


class AgentSendDtmf(BaseModel):
    type: Literal["agent_send_dtmf"] = "agent_send_dtmf"
    button: str


class LogMetric(BaseModel):
    type: Literal["log_metric"] = "log_metric"
    name: str
    value: Any


class LogMessage(BaseModel):
    type: Literal["log_message"] = "log_message"
    name: str
    level: Literal["info", "error"]
    message: str
    metadata: Optional[Dict[str, Any]] = None


class AgentUpdateCall(BaseModel):
    type: Literal["update_call"] = "update_call"
    voice_id: Optional[str] = None
    pronunciation_dict_id: Optional[str] = None


OutputEvent = Union[
    AgentSendText,
    AgentSendDtmf,
    AgentEndCall,
    AgentTransferCall,
    AgentToolCalled,
    AgentToolReturned,
    LogMetric,
    LogMessage,
    AgentUpdateCall,
]


# -------------------------
# Custom Events (agent-internal)
# -------------------------


class CustomHistoryEntry(BaseModel):
    """Custom text entry injected into history via add_history_entry.

    Not an InputEvent or OutputEvent — exists only in the agent's internal history
    and appears as a user message in the LLM conversation.
    """

    type: Literal["custom_history_entry"] = "custom_history_entry"
    content: str


# -------------------------
# Input Events (harness -> agent)
# -------------------------
# Each event has a stable event_id (UUID) for tracking which events trigger responses.
# history=None indicates the event is used within a history list (no nested history).


class CallStarted(BaseModel):
    type: Literal["call_started"] = "call_started"
    event_id: str = Field(default_factory=_generate_event_id)
    history: Optional[List["InputEvent"]] = None


class CallEnded(BaseModel):
    type: Literal["call_ended"] = "call_ended"
    event_id: str = Field(default_factory=_generate_event_id)
    history: Optional[List["InputEvent"]] = None


class AgentHandedOff(BaseModel):
    """Event emitted when control is transferred to the tool target."""

    type: Literal["agent_handed_off"] = "agent_handed_off"
    event_id: str = Field(default_factory=_generate_event_id)
    history: Optional[List["InputEvent"]] = None


class UserTurnStarted(BaseModel):
    type: Literal["user_turn_started"] = "user_turn_started"
    event_id: str = Field(default_factory=_generate_event_id)
    history: Optional[List["InputEvent"]] = None


class UserDtmfSent(BaseModel):
    type: Literal["user_dtmf_sent"] = "user_dtmf_sent"
    button: str
    event_id: str = Field(default_factory=_generate_event_id)
    history: Optional[List["InputEvent"]] = None


class UserTextSent(BaseModel):
    type: Literal["user_text_sent"] = "user_text_sent"
    content: str
    event_id: str = Field(default_factory=_generate_event_id)
    history: Optional[List["InputEvent"]] = None


class UserTurnEnded(BaseModel):
    type: Literal["user_turn_ended"] = "user_turn_ended"
    content: List[Union[UserDtmfSent, UserTextSent]] = Field(default_factory=list)
    event_id: str = Field(default_factory=_generate_event_id)
    history: Optional[List["InputEvent"]] = None


class AgentTurnStarted(BaseModel):
    type: Literal["agent_turn_started"] = "agent_turn_started"
    event_id: str = Field(default_factory=_generate_event_id)
    history: Optional[List["InputEvent"]] = None


class AgentTextSent(BaseModel):
    type: Literal["agent_text_sent"] = "agent_text_sent"
    content: str
    event_id: str = Field(default_factory=_generate_event_id)
    history: Optional[List["InputEvent"]] = None


class AgentDtmfSent(BaseModel):
    type: Literal["agent_dtmf_sent"] = "agent_dtmf_sent"
    button: str
    event_id: str = Field(default_factory=_generate_event_id)
    history: Optional[List["InputEvent"]] = None


class AgentTurnEnded(BaseModel):
    type: Literal["agent_turn_ended"] = "agent_turn_ended"
    content: List[
        Union[
            AgentTextSent,
            AgentDtmfSent,
        ]
    ] = Field(default_factory=list)
    event_id: str = Field(default_factory=_generate_event_id)
    history: Optional[List["InputEvent"]] = None


InputEvent = Union[
    CallStarted,
    UserTurnStarted,
    UserDtmfSent,
    UserTextSent,
    UserTurnEnded,
    AgentTurnStarted,
    AgentTextSent,
    AgentDtmfSent,
    AgentTurnEnded,
    AgentHandedOff,
    CallEnded,
]


# -------------------------
# History Events (used in LLM message building)
# -------------------------

HistoryEvent = Union[
    InputEvent,
    AgentToolCalled,
    AgentToolReturned,
    CustomHistoryEntry,
]


__all__ = [
    # Output
    "AgentSendText",
    "AgentSendDtmf",
    "AgentEndCall",
    "AgentTransferCall",
    "AgentToolCalled",
    "AgentToolReturned",
    "AgentHandedOff",
    "LogMetric",
    "LogMessage",
    "AgentUpdateCall",
    "OutputEvent",
    # Custom
    "CustomHistoryEntry",
    # Input
    "CallStarted",
    "CallEnded",
    "UserTurnStarted",
    "UserDtmfSent",
    "UserTextSent",
    "UserTurnEnded",
    "AgentTurnStarted",
    "AgentTextSent",
    "AgentDtmfSent",
    "AgentTurnEnded",
    "InputEvent",
    # History
    "HistoryEvent",
]
