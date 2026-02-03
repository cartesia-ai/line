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
    TEMPERATURE,
    prompt_agent1,
    prompt_agent2,
    prompt_agent3,
    schema_background,
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

    Awkwardness notes:
    1. LlmConfig doesn't have first-class response_format support - must use `extra`
    2. Need to construct a synthetic UserTurnEnded event just to pass history
    3. Agent produces AgentSendText events that we consume but don't yield
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
                # NOTE: This is awkward - LlmConfig doesn't have first-class response_format support
                extra={
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "analysis_schema",
                            "strict": True,
                            "schema": schema_background,
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
            # NOTE: This is awkward - we need to construct an event just to pass history
            synthetic_event = UserTurnEnded(
                content=[UserTextSent(content="Please analyze the interview so far.")],
                history=history,
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


# Pre-instantiated judges (created once, reused across calls)
_technical_judge: Optional[BackgroundJudge] = None
_communication_judge: Optional[BackgroundJudge] = None
_reasoning_judge: Optional[BackgroundJudge] = None


def _get_judges() -> tuple[BackgroundJudge, BackgroundJudge, BackgroundJudge]:
    """Lazy initialization of judge instances."""
    global _technical_judge, _communication_judge, _reasoning_judge
    if _technical_judge is None:
        _technical_judge = BackgroundJudge(prompt_agent1, "Technical Report")
        _communication_judge = BackgroundJudge(prompt_agent2, "Communication Report")
        _reasoning_judge = BackgroundJudge(prompt_agent3, "Reasoning Report")
    return _technical_judge, _communication_judge, _reasoning_judge


async def run_all_judges(env: TurnEnv, history: List):
    """Run all three judges in parallel."""
    technical, communication, reasoning = _get_judges()
    await asyncio.gather(
        technical.analyze(env, history),
        communication.analyze(env, history),
        reasoning.analyze(env, history),
        return_exceptions=True,
    )
