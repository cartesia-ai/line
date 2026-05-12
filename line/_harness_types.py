"""
Raw websocket message types

ConversationRunner maps from these to the "internal" InputEvent and OutputEvent types.
"""

from typing import Any, Dict, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

########################################################
#  Copied and adapted from Bifrost agent_types.py
########################################################

# Input messages to be sent over the websocket to the user code


class TranscriptionInput(BaseModel):
    content: str
    type: Literal["message"] = "message"
    event_id: Optional[str] = None


class DTMFInput(BaseModel):
    button: str
    type: Literal["dtmf"] = "dtmf"
    event_id: Optional[str] = None


class UserStateInput(BaseModel):
    value: str
    type: Literal["user_state"] = "user_state"
    event_id: Optional[str] = None


class AgentStateInput(BaseModel):
    value: str
    type: Literal["agent_state"] = "agent_state"
    event_id: Optional[str] = None


class ValidationErrorInput(BaseModel):
    error_message: str
    error_type: str
    type: Literal["validation_error"] = "validation_error"
    event_id: Optional[str] = None


class AgentSpeechInput(BaseModel):
    content: str
    type: Literal["agent_speech"] = "agent_speech"
    event_id: Optional[str] = None


class CustomInput(BaseModel):
    metadata: Dict[str, object]
    type: Literal["custom"] = "custom"
    event_id: Optional[str] = None


InputMessage = Union[
    TranscriptionInput,
    DTMFInput,
    UserStateInput,
    AgentStateInput,
    ValidationErrorInput,
    AgentSpeechInput,
    CustomInput,
]


# Output messages to be received from the user code


class ErrorOutput(BaseModel):
    type: Literal["error"] = "error"
    content: str


class DTMFOutput(BaseModel):
    type: Literal["dtmf"] = "dtmf"
    button: str
    responding_to: Optional[str] = None


class MessageOutput(BaseModel):
    type: Literal["message"] = "message"
    content: str
    interruptible: bool = True
    responding_to: Optional[str] = None


class ToolCallOutput(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    name: str
    arguments: Dict[str, object]
    result: Optional[str] = None
    id: Optional[str] = None
    responding_to: Optional[str] = None


class TransferOutput(BaseModel):
    type: Literal["transfer"] = "transfer"
    target_phone_number: str
    responding_to: Optional[str] = None
    interruptible: bool = True


class EndCallOutput(BaseModel):
    type: Literal["end_call"] = "end_call"
    responding_to: Optional[str] = None
    interruptible: bool = True


class LogEventOutput(BaseModel):
    type: Literal["log_event"] = "log_event"
    event: str
    metadata: Optional[Dict[str, object]] = None
    responding_to: Optional[str] = None


class LogMetricOutput(BaseModel):
    type: Literal["log_metric"] = "log_metric"
    name: str
    value: object
    responding_to: Optional[str] = None


class TTSConfig(BaseModel):
    voice_id: Optional[str] = None
    pronunciation_dict_id: Optional[str] = None
    language: Optional[str] = None


class STTConfig(BaseModel):
    language: Optional[str] = None


class ConfigOutput(BaseModel):
    type: Literal["config"] = "config"
    tts: Optional[TTSConfig] = None
    stt: Optional[STTConfig] = None
    language: Optional[str] = None
    responding_to: Optional[str] = None


class CustomOutput(BaseModel):
    type: Literal["custom"] = "custom"
    metadata: Dict[str, object]
    responding_to: Optional[str] = None


OutputMessage = Union[
    ErrorOutput,
    DTMFOutput,
    MessageOutput,
    ToolCallOutput,
    TransferOutput,
    EndCallOutput,
    LogEventOutput,
    LogMetricOutput,
    ConfigOutput,
    CustomOutput,
]


########################################################
#  Connection-level messages
#  These are handled during websocket setup, before
#  the conversation loop. Not part of InputMessage.
########################################################


class StartInput(BaseModel):
    """Start message sent by the harness with call parameters.

    Delivered once at connection start when the harness detects
    cartesia_version in the websocket URL. Carries the same call
    context that legacy clients pass via URL query params.
    """

    type: Literal["start"] = "start"
    call_id: str = "unknown"
    from_: str = Field(default="unknown", alias="from")
    to: str = "unknown"
    agent_call_id: str = "unknown"
    zdr: bool = False
    agent: Dict[str, Any] = Field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = None
    # Agent-scoped JWT minted by the API. Forwarded by the harness so agent
    # code can authenticate calls back to the API on behalf of the agent (e.g.
    # knowledge base document queries).
    agent_token: Optional[str] = None
    # Base URL for callbacks to the Cartesia API. Forwarded by the harness so
    # the agent uses the same API endpoint that minted its credentials,
    # rather than guessing via env var or a hardcoded prod default.
    api_base_url: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)
