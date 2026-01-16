"""
LLM wrapper module for the Line SDK.

This module provides a unified interface for working with 100+ LLM providers
via LiteLLM (OpenAI, Anthropic, Google, and many more) with a powerful tool
calling system supporting three paradigms:

- **Loopback tools**: Response loops back to the LLM for continued generation
- **Passthrough tools**: Response is emitted directly to the user
- **Handoff tools**: Control is transferred to another agent/handler

Quick Start:
    ```python
    from line.v02.llm import LlmAgent, LlmConfig, function_tool, passthrough_tool, Field
    from line.v02.llm import AgentSendText, AgentEndCall, TurnEnv, UserTextSent
    from typing import Annotated

    # Define a loopback tool (default)
    @function_tool
    async def get_weather(
        ctx: ToolContext,
        city: Annotated[str, Field(description="The city name")]
    ) -> str:
        '''Get the current weather in a city'''
        return f"72°F and sunny in {city}"

    # Define a passthrough tool
    @passthrough_tool
    async def end_call(
        ctx: ToolContext,
        message: Annotated[str, Field(description="Goodbye message")]
    ):
        '''End the call with a goodbye message'''
        yield AgentSendText(text=message)
        yield AgentEndCall()

    # Create the agent
    agent = LlmAgent(
        model="gpt-4o",  # LiteLLM format: "gpt-4o", "anthropic/claude-3-5-sonnet-20241022", etc.
        tools=[get_weather, end_call],
        config=LlmConfig(
            system_prompt="You are a helpful weather assistant.",
            introduction="Hello! I can help you check the weather.",
            temperature=0.7,
            num_retries=3,  # Built-in retry logic
            fallbacks=["anthropic/claude-3-5-sonnet-20241022"],  # Automatic fallback
        ),
    )

    # In get_agent
    def get_agent(ctx, call_request):
        return agent
    ```

Model Naming (LiteLLM format):
    - OpenAI: "gpt-4o", "gpt-4o-mini", "o1"
    - Anthropic: "anthropic/claude-3-5-sonnet-20241022", "anthropic/claude-3-opus-20240229"
    - Google: "gemini/gemini-2.0-flash", "gemini/gemini-1.5-pro"
    - See https://docs.litellm.ai/docs/providers for 100+ more providers
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
