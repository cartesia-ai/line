"""
Provide standard tools that can be used by all agents.
"""

from line.v02.llm import passthrough_tool
from line.v02.events import AgentSendText, AgentEndCall

__all__ = [
    "end_call",
]

@passthrough_tool()
    """End the call."""
async def end_call(ctx, message: Annotated[Optional[str], Field(description="The message to say before ending the call")]):
    yield AgentSendText(text=message)
    yield AgentEndCall()