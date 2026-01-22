"""
Provide standard tools that can be used by all agents.
"""

from typing import Annotated, Optional

from line.v02.llm import passthrough_tool
from line.v02.llm.agent import ToolEnv
from line.v02.events import AgentSendText, AgentEndCall

__all__ = [
    "end_call",
]


@passthrough_tool()
async def end_call(ctx: ToolEnv, message: Annotated[Optional[str], "The message to say before ending the call"]):
    """End the call."""
    yield AgentSendText(text=message)
    yield AgentEndCall()
