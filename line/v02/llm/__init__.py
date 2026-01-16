"""
LLM wrapper module for the Line SDK.

Provides a unified interface for 100+ LLM providers via LiteLLM with three
tool calling paradigms: loopback, passthrough, and handoff.

See README.md for examples and detailed documentation.
"""

# Agent types and events from v02
from line.v02.llm.agent import (
    # Agent protocol
    Agent,
    AgentCallable,
    AgentClass,
    # Input events with history
    AgentDTMFSent,
    # Output events (v02)
    AgentEndCall,
    # LLM-specific events
    AgentHandoff,
    AgentSendDTMF,
    AgentSendText,
    AgentSpec,
    AgentTextSent,
    AgentToolCalled,
    AgentToolReturned,
    AgentTransferCall,
    AgentTurnEnded,
    AgentTurnStarted,
    CallEnded,
    CallStarted,
    EventFilter,
    InputEvent,
    LogMessage,
    LogMetric,
    OutputEvent,
    # Specific events (used in history lists)
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
    ToolCallEvent,
    ToolResultEvent,
    TurnEnv,
    UserDtmfSent,
    UserTextSent,
    UserTurnEnded,
    UserTurnStarted,
)

# Configuration
from line.v02.llm.config import LlmConfig

# Function tool definitions
from line.v02.llm.function_tool import Field, FunctionTool, ToolType, function_tool

# LLM Agent
from line.v02.llm.llm_agent import LlmAgent

# Provider base classes
from line.v02.llm.providers import LLM, LLMStream, Message, StreamChunk, ToolCall, UsageInfo

# Schema converters
from line.v02.llm.schema_converter import (
    function_tool_to_anthropic,
    function_tool_to_gemini,
    function_tool_to_openai,
    function_tools_to_anthropic,
    function_tools_to_gemini,
    function_tools_to_openai,
    merge_gemini_tools,
)

# Tool context
from line.v02.llm.tool_context import ToolContext, ToolResult

# Tool type decorators
from line.v02.llm.tool_types import handoff_tool, loopback_tool, passthrough_tool

__all__ = [
    # LLM Agent
    "LlmAgent",
    # Agent protocol
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
    "AgentHandoff",
    "ToolCallEvent",
    "ToolResultEvent",
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
    # Configuration
    "LlmConfig",
    # Tool definitions
    "Field",
    "FunctionTool",
    "ToolType",
    "function_tool",
    # Tool type decorators
    "loopback_tool",
    "passthrough_tool",
    "handoff_tool",
    # Tool context
    "ToolContext",
    "ToolResult",
    # Schema converters
    "function_tool_to_openai",
    "function_tool_to_anthropic",
    "function_tool_to_gemini",
    "function_tools_to_openai",
    "function_tools_to_anthropic",
    "function_tools_to_gemini",
    "merge_gemini_tools",
    # Provider base classes
    "LLM",
    "LLMStream",
    "Message",
    "StreamChunk",
    "ToolCall",
    "UsageInfo",
]
