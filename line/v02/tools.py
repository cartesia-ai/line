"""
Provide standard tools that can be used by all agents.
"""

from line.v02.llm import passthrough_tool
from line.v02.events import AgentSendText, AgentEndCall
from typing import Annotated, Optional
from line.v02.llm.function_tool import Field

__all__ = [
    "end_call",
]

@passthrough_tool()
async def end_call(ctx, message: Annotated[Optional[str], Field(description="The message to say before ending the call")]):
    """End the call."""
    yield AgentSendText(text=message)
    yield AgentEndCall()