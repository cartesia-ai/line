"""
LLM wrapper module for the Line SDK.

This module provides a unified interface for working with different LLM providers
(OpenAI, Anthropic, Google) and a powerful tool calling system with three paradigms:

- **Loopback tools**: Response loops back to the LLM for continued generation
- **Passthrough tools**: Response is emitted directly to the user
- **Handoff tools**: Control is transferred to another agent/handler

Quick Start:
    ```python
    from line.llm import LlmAgent, LlmConfig, function_tool, passthrough_tool, Field
    from line.llm import AgentOutput, AgentEndCall, TurnContext, UserTranscript
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
        yield AgentOutput(text=message)
        yield AgentEndCall()

    # Create the agent
    agent = LlmAgent(
        model="gemini-2.0-flash",  # or "gpt-4o", "claude-3.5-sonnet"
        api_key=os.getenv("GEMINI_API_KEY"),
        tools=[get_weather, end_call],
        config=LlmConfig(
            system_prompt="You are a helpful weather assistant.",
            introduction="Hello! I can help you check the weather.",
            temperature=0.7,
        ),
    )

    # In get_agent
    def get_agent(ctx, call_request):
        return agent
    ```

Supported Models:
    - OpenAI: gpt-4o, gpt-4o-mini, gpt-4-turbo, o1, o1-mini
    - Anthropic: claude-3.5-sonnet, claude-3.5-haiku, claude-3-opus, opus-4.5
    - Google: gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash
"""

# Agent types and events
from line.llm.agent import (
    # Input events
    InputEvent,
    CallStarted,
    CallEnded,
    UserTurnStarted,
    UserTurnEnded,
    UserTranscript,
    UserDTMF,
    # Output events
    OutputEvent,
    AgentOutput,
    AgentEndTurn,
    AgentEndCall,
    AgentTransferCall,
    AgentSendDTMF,
    AgentHandoff,
    ToolCallOutput,
    ToolResultOutput,
    # Context
    TurnContext,
    # Agent protocol
    Agent,
    AgentClass,
    AgentFunction,
    # Predicates
    TurnStartPredicate,
    TurnEndPredicate,
    DEFAULT_TURN_START_EVENTS,
    DEFAULT_TURN_END_EVENTS,
    make_predicate,
)

# Configuration
from line.llm.config import LlmConfig, MODEL_ALIASES, detect_provider, resolve_model_alias

# Function tool definitions
from line.llm.function_tool import Field, FunctionTool, ToolType, function_tool

# Tool type decorators
from line.llm.tool_types import handoff_tool, loopback_tool, passthrough_tool

# Tool context
from line.llm.tool_context import ToolContext, ToolResult

# Schema converters
from line.llm.schema_converter import (
    function_tool_to_anthropic,
    function_tool_to_gemini,
    function_tool_to_openai,
    function_tools_to_anthropic,
    function_tools_to_gemini,
    function_tools_to_openai,
    merge_gemini_tools,
)

# LLM Agent
from line.llm.llm_agent import LlmAgent

# Provider base classes
from line.llm.providers import LLM, LLMStream, Message, StreamChunk, ToolCall, UsageInfo

__all__ = [
    # LLM Agent
    "LlmAgent",
    # Agent protocol
    "Agent",
    "AgentClass",
    "AgentFunction",
    # Input events
    "InputEvent",
    "CallStarted",
    "CallEnded",
    "UserTurnStarted",
    "UserTurnEnded",
    "UserTranscript",
    "UserDTMF",
    # Output events
    "OutputEvent",
    "AgentOutput",
    "AgentEndTurn",
    "AgentEndCall",
    "AgentTransferCall",
    "AgentSendDTMF",
    "AgentHandoff",
    "ToolCallOutput",
    "ToolResultOutput",
    # Context
    "TurnContext",
    # Predicates
    "TurnStartPredicate",
    "TurnEndPredicate",
    "DEFAULT_TURN_START_EVENTS",
    "DEFAULT_TURN_END_EVENTS",
    "make_predicate",
    # Configuration
    "LlmConfig",
    "MODEL_ALIASES",
    "detect_provider",
    "resolve_model_alias",
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
