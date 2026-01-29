"""
Tools module for LLM agents.

Re-exports decorators, system tools, and utility types.
"""

# Decorators
from line.v02.llm_agent.tools.decorators import (
    handoff_tool,
    loopback_tool,
    passthrough_tool,
)

# System tools
from line.v02.llm_agent.tools.system import (
    DtmfButton,
    WebSearchTool,
    end_call,
    send_dtmf,
    transfer_call,
    web_search,
)

# Utility types
from line.v02.llm_agent.tools.utils import (
    FunctionTool,
    HandoffToolFn,
    LoopbackToolFn,
    ParameterInfo,
    PassthroughToolFn,
    ToolEnv,
    ToolType,
    construct_function_tool,
)

__all__ = [
    # Decorators
    "loopback_tool",
    "passthrough_tool",
    "handoff_tool",
    # System tools
    "DtmfButton",
    "WebSearchTool",
    "web_search",
    "end_call",
    "send_dtmf",
    "transfer_call",
    # Utility types
    "ToolType",
    "ToolEnv",
    "LoopbackToolFn",
    "PassthroughToolFn",
    "HandoffToolFn",
    "FunctionTool",
    "ParameterInfo",
    "construct_function_tool",
]
