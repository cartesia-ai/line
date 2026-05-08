"""
LLM Agent module for the Line SDK.

Provides a unified interface for 100+ LLM providers via LiteLLM. Tool calls
either feed back to the LLM (raw values) or pass directly through to the user
(OutputEvent yields); handoff tools transfer control to another agent.

See README.md for examples and detailed documentation.
"""

# Configuration
from line.llm_agent.config import FALLBACK_INTRODUCTION, FALLBACK_SYSTEM_PROMPT, LlmConfig

# History
from line.llm_agent.history import History

# LLM Agent
from line.llm_agent.llm_agent import LlmAgent

# Provider facade
from line.llm_agent.provider import ChatStream, LLMProvider, LlmProvider

# Tool decorators
from line.llm_agent.tools.decorators import handoff_tool, loopback_tool, passthrough_tool

# Built-in tools
from line.llm_agent.tools.system import (
    agent_as_handoff,
    end_call,
    knowledge_base,
    mcp_tool,
    send_dtmf,
    transfer_call,
    web_search,
)

# Function tool definitions and types
from line.llm_agent.tools.utils import (
    FunctionTool,
    HandoffToolFn,
    LoopbackToolFn,
    ParameterInfo,
    PassthroughToolFn,
    ToolEnv,
    ToolType,
)

__all__ = [
    # History
    "History",
    # LLM Agent
    "LlmAgent",
    # Provider
    "ChatStream",
    "LlmProvider",
    "LLMProvider",
    # Configuration
    "LlmConfig",
    "FALLBACK_SYSTEM_PROMPT",
    "FALLBACK_INTRODUCTION",
    # Tool decorators
    "loopback_tool",
    "passthrough_tool",
    "handoff_tool",
    # Built-in tools
    "end_call",
    "send_dtmf",
    "transfer_call",
    "web_search",
    "agent_as_handoff",
    "mcp_tool",
    "knowledge_base",
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
