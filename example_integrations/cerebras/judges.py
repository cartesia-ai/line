"""
Background Judge Agents using LlmAgent with Cerebras via LiteLLM.

These agents analyze interview responses in the background without speaking to the user.
They evaluate technical expertise, communication skills, and reasoning ability.

Note: This implementation intentionally uses LlmAgent (rather than litellm.acompletion directly)
to surface any API friction for future SDK improvements.
"""

import asyncio
import os
from typing import List, Optional

from config import (
    MODEL_ID_BACK,
    PROMPT_AGENT1,
    PROMPT_AGENT2,
    PROMPT_AGENT3,
    SCHEMA_BACKGROUND,
    TEMPERATURE,
)
from loguru import logger
from pydantic import BaseModel, Field
from report_logger import SimpleLogger

from line.agent import TurnEnv
from line.events import AgentSendText, UserTextSent, UserTurnEnded
from line.llm_agent import LlmAgent, LlmConfig


class EvalInfo(BaseModel):
    """Schema for extracted evaluation information."""

    competence: str = Field(..., description="Competence level: HIGH, MEDIUM, or LOW")
    strengths: str = Field(..., description="Identified strengths")
    weaknesses: str = Field(..., description="Identified weaknesses")


class BackgroundJudge:
    """
    Background judge that uses LlmAgent to analyze interview responses.

    Note: This is intentionally using LlmAgent (rather than litellm directly)
    to surface any API friction for future SDK improvements.

    """

    def __init__(self, system_prompt: str, node_name: str):
        self._node_name = node_name
        self._logger = SimpleLogger(node_name)
        self._agent = LlmAgent(
            model=MODEL_ID_BACK,
            api_key=os.getenv("CEREBRAS_API_KEY"),
            config=LlmConfig(
                system_prompt=system_prompt,
                temperature=TEMPERATURE,
                max_tokens=50,
                # Pass structured output schema via extra
                extra={
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "analysis_schema",
                            "strict": True,
                            "schema": SCHEMA_BACKGROUND,
                        },
                    }
                },
            ),
        )

    async def analyze(self, env: TurnEnv, history: List) -> Optional[EvalInfo]:
        """
        Run analysis on the conversation history.

        Creates a synthetic UserTurnEnded event to trigger the LlmAgent,
        then consumes (but doesn't yield) the output events.
        """
        try:
            # Get the latest user response for logging
            latest_response = ""
            for event in reversed(history or []):
                if isinstance(event, UserTextSent):
                    latest_response = event.content
                    break

            # Create synthetic event with history
            synthetic_event_without_history = UserTurnEnded(
                content=[UserTextSent(content="Please analyze the interview so far.")],
            )
            synthetic_event = UserTurnEnded(
                **synthetic_event_without_history.model_dump(),
                history=history + [synthetic_event_without_history],
            )

            # Collect output text (consume but don't yield events)
            full_text = ""
            async for output in self._agent.process(env, synthetic_event):
                if isinstance(output, AgentSendText):
                    full_text += output.text

            # Parse as structured output
            if full_text:
                eval_info = EvalInfo.model_validate_json(full_text)

                # Log to file
                self._logger._write(f"\n[RESPONSE]\n{latest_response}\n")
                self._logger._write("-" * 40 + "\n")
                self._logger._write(f"{self._node_name}:\n")
                self._logger._write(f"  competence: {eval_info.competence}\n")
                self._logger._write(f"  strengths: {eval_info.strengths}\n")
                self._logger._write(f"  weaknesses: {eval_info.weaknesses}\n")
                self._logger._write("-" * 40 + "\n")

                logger.info(f"Judge {self._node_name}: {eval_info.model_dump_json()}")
                return eval_info

        except Exception as e:
            logger.exception(f"Judge {self._node_name} failed: {e}")
        return None

    async def cleanup(self):
        await self._agent.cleanup()


def create_judges() -> tuple["BackgroundJudge", "BackgroundJudge", "BackgroundJudge"]:
    """
    Create a fresh set of judge instances for a new call.

    Each call should have its own judges to avoid state mixing between
    concurrent interviews (LlmAgent maintains per-conversation state).
    """
    return (
        BackgroundJudge(PROMPT_AGENT1, "Technical Report"),
        BackgroundJudge(PROMPT_AGENT2, "Communication Report"),
        BackgroundJudge(PROMPT_AGENT3, "Reasoning Report"),
    )


async def run_all_judges(
    judges: tuple["BackgroundJudge", "BackgroundJudge", "BackgroundJudge"],
    env: TurnEnv,
    history: List,
):
    """Run all three judges in parallel."""
    technical, communication, reasoning = judges
    await asyncio.gather(
        technical.analyze(env, history),
        communication.analyze(env, history),
        reasoning.analyze(env, history),
        return_exceptions=True,
    )
