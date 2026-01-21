"""
LLM wrapper module for the Line SDK.

Provides a unified interface for 100+ LLM providers via LiteLLM with three
tool calling paradigms: loopback, passthrough, and handoff.

See README.md for examples and detailed documentation.
"""

# Agent types, events, and tool types
from line.v02.llm.agent import (
    Agent,
    AgentCallable,
    AgentClass,
    AgentDTMFSent,
    AgentEndCall,
    AgentHandedOff,
    AgentSendDTMF,
    AgentSendText,
    AgentSpec,
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
    EventFilter,
    HandoffToolFn,
    InputEvent,
    LogMessage,
    LogMetric,
    LoopbackToolFn,
    OutputEvent,
    PassthroughToolFn,
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
    ToolEnv,
    TurnEnv,
    UserDtmfSent,
    UserTextSent,
    UserTurnEnded,
    UserTurnStarted,
)

# Configuration
from line.v02.llm.config import LlmConfig

# Function tool definitions
from line.v02.llm.function_tool import Field, FunctionTool, ToolType, construct_function_tool

# LLM Agent
from line.v02.llm.llm_agent import LlmAgent

# Provider
from line.v02.llm.provider import LLMProvider, Message, StreamChunk, ToolCall

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

# Tool type decorators
from line.v02.llm.tool_types import handoff_tool, loopback_tool, passthrough_tool

__all__ = [
    # LLM Agent
    "LlmAgent",
    # Configuration
    "LlmConfig",
    # Tool definitions
    "Field",
    "FunctionTool",
    "ToolType",
    "construct_function_tool",
    # Tool type decorators
    "loopback_tool",
    "passthrough_tool",
    "handoff_tool",
    # Schema converters
    "function_tool_to_openai",
    "function_tool_to_anthropic",
    "function_tool_to_gemini",
    "function_tools_to_openai",
    "function_tools_to_anthropic",
    "function_tools_to_gemini",
    "merge_gemini_tools",
    # Provider
    "LLMProvider",
    "Message",
    "StreamChunk",
    "ToolCall",
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
