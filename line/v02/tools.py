"""
Provide standard tools that can be used by all agents.
"""

from line.v02.llm import passthrough_tool
from line.v02.events import AgentSendText, AgentEndCall

__all__ = [
    "end_call",
]

@passthrough_tool()
async def end_call(ctx, message: str = "Goodbye!"):
    """End the call."""
    yield AgentSendText(text=message)
    yield AgentEndCall()