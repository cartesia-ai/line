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


class DTMFInput(BaseModel):
    button: str
    type: Literal["dtmf"] = "dtmf"


class UserStateInput(BaseModel):
    value: str
    type: Literal["user_state"] = "user_state"


class AgentStateInput(BaseModel):
    value: str
    type: Literal["agent_state"] = "agent_state"


class ValidationErrorInput(BaseModel):
    error_message: str
    error_type: str
    type: Literal["validation_error"] = "validation_error"


class AgentSpeechInput(BaseModel):
    content: str
    type: Literal["agent_speech"] = "agent_speech"


class CustomInput(BaseModel):
    metadata: Dict[str, object]
    type: Literal["custom"] = "custom"


class StartInput(BaseModel):
    """Start message sent by the external service with call parameters."""

    type: Literal["start"] = "start"
    call_id: str = "unknown"
    from_: str = Field(default="unknown", alias="from")
    to: str = "unknown"
    agent_call_id: str = "unknown"
    agent: Dict[str, Any] = Field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(populate_by_name=True)


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


class MessageOutput(BaseModel):
    type: Literal["message"] = "message"
    content: str
    interruptible: bool = True


class ToolCallOutput(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    name: str
    arguments: Dict[str, object]
    result: Optional[str] = None
    id: Optional[str] = None


class TransferOutput(BaseModel):
    type: Literal["transfer"] = "transfer"
    target_phone_number: str


class EndCallOutput(BaseModel):
    type: Literal["end_call"] = "end_call"


class LogEventOutput(BaseModel):
    type: Literal["log_event"] = "log_event"
    event: str
    metadata: Optional[Dict[str, object]] = None


class LogMetricOutput(BaseModel):
    type: Literal["log_metric"] = "log_metric"
    name: str
    value: object


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


class CustomOutput(BaseModel):
    type: Literal["custom"] = "custom"
    metadata: Dict[str, object]


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
