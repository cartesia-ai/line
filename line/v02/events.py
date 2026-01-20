"""Typed event definitions for the v0.2 audio harness."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

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


class AgentSendDTMF(BaseModel):
    type: Literal["agent_send_dtmf"] = "agent_send_dtmf"
    button: str


class AgentHandedOff(BaseModel):
    """Event emitted when control is transferred to the tool target."""

    type: Literal["agent_handed_off"] = "agent_handed_off"


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


OutputEvent = Union[
    AgentSendText,
    AgentSendDTMF,
    AgentEndCall,
    AgentTransferCall,
    AgentToolCalled,
    AgentToolReturned,
    LogMetric,
    LogMessage,
]


# -------------------------
# Input Events (harness -> agent)
# -------------------------
# Specific* events do NOT include history and are used within the history list.


class SpecificCallStarted(BaseModel):
    type: Literal["call_started"] = "call_started"


class SpecificCallEnded(BaseModel):
    type: Literal["call_ended"] = "call_ended"


class SpecificUserTurnStarted(BaseModel):
    type: Literal["user_turn_started"] = "user_turn_started"


class SpecificUserDtmfSent(BaseModel):
    type: Literal["user_dtmf_sent"] = "user_dtmf_sent"
    button: str


class SpecificUserTextSent(BaseModel):
    type: Literal["user_text_sent"] = "user_text_sent"
    content: str


class SpecificUserTurnEnded(BaseModel):
    type: Literal["user_turn_ended"] = "user_turn_ended"
    content: List[Union[SpecificUserDtmfSent, SpecificUserTextSent]] = Field(default_factory=list)


class SpecificAgentTurnStarted(BaseModel):
    type: Literal["agent_turn_started"] = "agent_turn_started"


class SpecificAgentTextSent(BaseModel):
    type: Literal["agent_text_sent"] = "agent_text_sent"
    content: str


class SpecificAgentDTMFSent(BaseModel):
    type: Literal["agent_dtmf_sent"] = "agent_dtmf_sent"
    button: str


class SpecificAgentToolCalled(BaseModel):
    type: Literal["agent_tool_called"] = "agent_tool_called"
    tool_call_id: str
    tool_name: str
    tool_args: Dict[str, Any] = Field(default_factory=dict)


class SpecificAgentToolReturned(BaseModel):
    type: Literal["agent_tool_returned"] = "agent_tool_returned"
    tool_call_id: str
    tool_name: str
    tool_args: Dict[str, Any] = Field(default_factory=dict)
    result: Any = None


class SpecificAgentTurnEnded(BaseModel):
    type: Literal["agent_turn_ended"] = "agent_turn_ended"
    content: List[
        Union[
            SpecificAgentTextSent,
            SpecificAgentDTMFSent,
            SpecificAgentToolCalled,
            SpecificAgentToolReturned,
        ]
    ] = Field(default_factory=list)


SpecificInputEvent = Union[
    SpecificCallStarted,
    SpecificUserTurnStarted,
    SpecificUserDtmfSent,
    SpecificUserTextSent,
    SpecificUserTurnEnded,
    SpecificAgentTurnStarted,
    SpecificAgentTextSent,
    SpecificAgentDTMFSent,
    SpecificAgentToolCalled,
    SpecificAgentToolReturned,
    SpecificAgentTurnEnded,
    SpecificCallEnded,
    AgentToolCalled,
    AgentToolReturned,
]


class CallStarted(SpecificCallStarted):
    history: List[SpecificInputEvent] = Field(default_factory=list)


class CallEnded(SpecificCallEnded):
    history: List[SpecificInputEvent] = Field(default_factory=list)


class UserTurnStarted(SpecificUserTurnStarted):
    history: List[SpecificInputEvent] = Field(default_factory=list)


class UserDtmfSent(SpecificUserDtmfSent):
    history: List[SpecificInputEvent] = Field(default_factory=list)


class UserTextSent(SpecificUserTextSent):
    history: List[SpecificInputEvent] = Field(default_factory=list)


class UserTurnEnded(SpecificUserTurnEnded):
    history: List[SpecificInputEvent] = Field(default_factory=list)


class AgentTurnStarted(SpecificAgentTurnStarted):
    history: List[SpecificInputEvent] = Field(default_factory=list)


class AgentTextSent(SpecificAgentTextSent):
    history: List[SpecificInputEvent] = Field(default_factory=list)


class AgentDTMFSent(SpecificAgentDTMFSent):
    history: List[SpecificInputEvent] = Field(default_factory=list)


class AgentToolCalledInput(AgentToolCalled):
    history: List[SpecificInputEvent] = Field(default_factory=list)


class AgentToolReturnedInput(AgentToolReturned):
    history: List[SpecificInputEvent] = Field(default_factory=list)


class AgentTurnEnded(SpecificAgentTurnEnded):
    history: List[SpecificInputEvent] = Field(default_factory=list)


InputEvent = Union[
    CallStarted,
    UserTurnStarted,
    UserDtmfSent,
    UserTextSent,
    UserTurnEnded,
    AgentTurnStarted,
    AgentTextSent,
    AgentDTMFSent,
    AgentToolCalledInput,
    AgentToolReturnedInput,
    AgentTurnEnded,
    CallEnded,
]


__all__ = [
    # Output
    "AgentSendText",
    "AgentSendDTMF",
    "AgentEndCall",
    "AgentTransferCall",
    "AgentToolCalled",
    "AgentToolReturned",
    "LogMetric",
    "LogMessage",
    "OutputEvent",
    # Input specific
    "SpecificCallStarted",
    "SpecificCallEnded",
    "SpecificUserTurnStarted",
    "SpecificUserDtmfSent",
    "SpecificUserTextSent",
    "SpecificUserTurnEnded",
    "SpecificAgentTurnStarted",
    "SpecificAgentTextSent",
    "SpecificAgentDTMFSent",
    "SpecificAgentToolCalled",
    "SpecificAgentToolReturned",
    "SpecificAgentTurnEnded",
    "SpecificInputEvent",
    # Input with history
    "CallStarted",
    "CallEnded",
    "UserTurnStarted",
    "UserDtmfSent",
    "UserTextSent",
    "UserTurnEnded",
    "AgentTurnStarted",
    "AgentTextSent",
    "AgentDTMFSent",
    "AgentToolCalledInput",
    "AgentToolReturnedInput",
    "AgentTurnEnded",
    "InputEvent",
]
