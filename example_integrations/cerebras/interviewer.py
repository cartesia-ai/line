"""
Interview Agent using LlmAgent with Cerebras via LiteLLM.

This is the main conversation agent that speaks to users during mock interviews.
It wraps an LlmAgent and coordinates with background judge agents.
"""

import asyncio
import os
from typing import Annotated, AsyncIterable

from config import INTRODUCTION, MAX_OUTPUT_TOKENS, MODEL_ID, TEMPERATURE, prompt_main
from judges import create_judges, run_all_judges
from loguru import logger

from line.agent import AgentClass, TurnEnv
from line.events import CallEnded, InputEvent, OutputEvent, UserTurnEnded
from line.llm_agent import LlmAgent, LlmConfig, ToolEnv, end_call, loopback_tool


class InterviewAgent(AgentClass):
    """
    Main interview agent that conducts mock interviews.

    Uses LlmAgent internally for conversation, with a wrapper to:
    1. Track interview state (started/not started)
    2. Trigger background judge analysis on each user turn
    """

    def __init__(self):
        self._interview_started = False

        # Create per-call judge instances
        self._judges = create_judges()

        # Main LlmAgent for conversation
        self._agent = LlmAgent(
            model=MODEL_ID,
            api_key=os.getenv("CEREBRAS_API_KEY"),
            tools=[end_call, self.start_interview],
            config=LlmConfig(
                system_prompt=prompt_main,
                introduction=INTRODUCTION,
                temperature=TEMPERATURE,
                max_tokens=MAX_OUTPUT_TOKENS,
            ),
        )

    @loopback_tool
    async def start_interview(
        self,
        ctx: ToolEnv,
        confirmed: Annotated[bool, "Set to true when user confirms ready to start the interview"],
    ) -> str:
        """
        Starts the interview after user confirmation.

        Call this when the user says they are ready to begin the interview.
        """
        self._interview_started = confirmed
        logger.info(f"Interview started: {self._interview_started}")

        if confirmed:
            return "Interview started. Ask the first interview question based on the role they mentioned."
        return "User declined to start the interview."

    async def process(self, env: TurnEnv, event: InputEvent) -> AsyncIterable[OutputEvent]:
        """Process input events and yield output events."""

        # Cleanup on call end
        if isinstance(event, CallEnded):
            await self._agent.cleanup()
            for judge in self._judges:
                await judge.cleanup()
            return

        # Fire background judges on each user turn (if interview started)
        if isinstance(event, UserTurnEnded) and self._interview_started:
            asyncio.create_task(run_all_judges(self._judges, env, event.history))

        # Delegate to LlmAgent for conversation
        async for output in self._agent.process(env, event):
            yield output

