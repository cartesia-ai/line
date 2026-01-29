"""
LLM Agent module for the Line SDK.

Provides a unified interface for 100+ LLM providers via LiteLLM with three
tool calling paradigms: loopback, passthrough, and handoff.

See README.md for examples and detailed documentation.
"""

# Configuration
from line.v02.llm_agent.config import FALLBACK_INTRODUCTION, FALLBACK_SYSTEM_PROMPT, LlmConfig

# LLM Agent
from line.v02.llm_agent.llm_agent import LlmAgent

# Tool type decorators
from line.v02.llm_agent.tools.decorators import handoff_tool, loopback_tool, passthrough_tool

# Function tool definitions and types
from line.v02.llm_agent.tools.utils import (
    FunctionTool,
    HandoffToolFn,
    LoopbackToolFn,
    ParameterInfo,
    PassthroughToolFn,
    ToolEnv,
    ToolType,
)

# Built-in tools
from line.v02.llm_agent.tools.system import end_call, send_dtmf, transfer_call, web_search, agent_as_handoff

__all__ = [
    # LLM Agent
    "LlmAgent",
    # Configuration
    "LlmConfig",
    "FALLBACK_SYSTEM_PROMPT",
    "FALLBACK_INTRODUCTION",
    # Tool type decorators
    "loopback_tool",
    "passthrough_tool",
    "handoff_tool",
    # Built-in tools
    "end_call",
    "send_dtmf",
    "transfer_call",
    "web_search",
    "agent_as_handoff",
    # Tool types
    "ToolEnv",
    "LoopbackToolFn",
    "PassthroughToolFn",
    "HandoffToolFn",
    # Tool definitions
    "FunctionTool",
    "ParameterInfo",
    "ToolType",
]
