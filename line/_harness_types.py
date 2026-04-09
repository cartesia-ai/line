"""
Raw websocket message types

ConversationRunner maps from these to the "internal" InputEvent and OutputEvent types.
"""

from typing import Dict, Literal, Optional, Union

from pydantic import BaseModel

########################################################
#  Copied and adapted from Bifrost agent_types.py
########################################################

# Input messages to be sent over the websocket to the user code


class TranscriptionInput(BaseModel):
    content: str
    type: Literal["message"] = "message"
    event_id: Union[int, str, None] = None
    turn_id: Union[int, str, None] = None


class DTMFInput(BaseModel):
    button: str
    type: Literal["dtmf"] = "dtmf"
    event_id: Union[int, str, None] = None
    turn_id: Union[int, str, None] = None


class UserStateInput(BaseModel):
    value: str
    type: Literal["user_state"] = "user_state"
    event_id: Union[int, str, None] = None
    turn_id: Union[int, str, None] = None


class AgentStateInput(BaseModel):
    value: str
    type: Literal["agent_state"] = "agent_state"
    event_id: Union[int, str, None] = None
    turn_id: Union[int, str, None] = None


class ValidationErrorInput(BaseModel):
    error_message: str
    error_type: str
    type: Literal["validation_error"] = "validation_error"


class AgentSpeechInput(BaseModel):
    content: str
    type: Literal["agent_speech"] = "agent_speech"
    event_id: Union[int, str, None] = None
    turn_id: Union[int, str, None] = None


class CustomInput(BaseModel):
    metadata: Dict[str, object]
    type: Literal["custom"] = "custom"
    event_id: Union[int, str, None] = None
    turn_id: Union[int, str, None] = None


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


class EndCallOutput(BaseModel):
    type: Literal["end_call"] = "end_call"
    responding_to: Optional[str] = None


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
