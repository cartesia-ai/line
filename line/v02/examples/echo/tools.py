from typing import Annotated

from loguru import logger

from line.v02.events import AgentHandedOff, AgentSendText, SpecificUserTextSent, UserTurnEnded
from line.v02.llm import handoff_tool
from line.v02.llm.agent import ToolEnv


@handoff_tool
async def echo(ctx: ToolEnv, prefix: Annotated[str, "A prefix to add before each echoed message"], event):
    """Echo the user's message back to them with a prefix."""
    if isinstance(event, AgentHandedOff):
        yield AgentSendText(text=f"Echo mode activated! I'll prefix everything with '{prefix}'")
        return

    if isinstance(event, UserTurnEnded):
        logger.info(f"Tool call echo: User turn ended: {event.content}")
        for item in event.content:
            if isinstance(item, SpecificUserTextSent):
                logger.info(f"Tool call echo: Echoing message: {prefix}: {item.content}")
                yield AgentSendText(text=f"{prefix}: {item.content}")
