"""
Provide standard tools that can be used by all agents.
"""

from typing import Annotated, Optional

from line.v02.events import AgentEndCall, AgentSendText
from line.v02.llm import passthrough_tool
from line.v02.llm.agent import ToolEnv

__all__ = [
    "end_call",
]


@passthrough_tool()
async def end_call(
    ctx: ToolEnv, 
    message: Annotated[[str], "The message to say before ending the call"] = None
):
    """End the call."""
    if message is not None:
        yield AgentSendText(text=message)
    yield AgentEndCall()
