import warnings

# DeprecationWarning is filtered out by default outside __main__, so it would
# never reach the people who need to see this.
warnings.warn(
    "The Cartesia Line SDK is deprecated. Cartesia stops hosting Line agents on "
    "2026-11-20, after which they will no longer take calls. Migrate to Cartesia "
    "managed agents: https://docs.cartesia.ai/agents",
    UserWarning,
    stacklevel=2,
)

# Core voice agent components
# Agent types
from line.agent import (
    Agent,
    AgentCallable,
    AgentClass,
    AgentSpec,
    EventFilter,
    TurnEnv,
)

# Events
from line.events import (
    AgentDtmfSent,
    AgentEndCall,
    AgentHandedOff,
    AgentSendCustom,
    AgentSendDtmf,
    # Output events
    AgentSendText,
    AgentTextSent,
    AgentToolCalled,
    AgentToolReturned,
    AgentTransferCall,
    AgentTurnEnded,
    AgentTurnStarted,
    AgentUpdateCall,
    CallEnded,
    # Input events
    CallStarted,
    # Custom events
    CustomHistoryEntry,
    EndCallReason,
    HistoryEvent,
    InputEvent,
    LogMessage,
    LogMetric,
    OutputEvent,
    UserCustomSent,
    UserDtmfSent,
    UserTextSent,
    UserTurnEnded,
    UserTurnStarted,
)
from line.knowledge_base import KnowledgeBase, KnowledgeBaseError
from line.voice_agent_app import (
    AgentConfig,
    AgentEnv,
    CallRequest,
    ConversationRunner,
    PreCallResult,
    VoiceAgentApp,
)
from line.word_buffer import word_buffer

__all__ = [
    # Voice Agent App
    "VoiceAgentApp",
    "ConversationRunner",
    "AgentEnv",
    "CallRequest",
    "AgentConfig",
    "PreCallResult",
    # Agent types
    "Agent",
    "AgentCallable",
    "AgentClass",
    "AgentSpec",
    "EventFilter",
    "TurnEnv",
    # Output events
    "AgentSendText",
    "AgentSendDtmf",
    "AgentEndCall",
    "EndCallReason",
    "AgentTransferCall",
    "AgentToolCalled",
    "AgentToolReturned",
    "LogMetric",
    "LogMessage",
    "AgentUpdateCall",
    "AgentSendCustom",
    "OutputEvent",
    # Input events
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
    "AgentHandedOff",
    "UserCustomSent",
    "InputEvent",
    # Custom events
    "CustomHistoryEntry",
    # History
    "HistoryEvent",
    # Knowledge base
    "KnowledgeBase",
    "KnowledgeBaseError",
    # Utilities
    "word_buffer",
]
