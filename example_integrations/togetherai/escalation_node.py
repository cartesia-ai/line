import json
from typing import AsyncGenerator

import config
from loguru import logger
from openai_utils import convert_messages_to_openai, format_escalation_report
from pydantic import BaseModel, Field

from line.events import AgentResponse
from line.nodes.conversation_context import ConversationContext
from line.nodes.reasoning import ReasoningNode


class EscalationInfo(BaseModel):
    """Schema for escalation analysis results."""

    escalation_level: str = Field(..., description="escalation level")
    reason: str = Field(..., description="reason for escalation")
    recommended_action: str = Field(..., description="recommended action")


class EscalationNode(ReasoningNode):
    """
    Background node that monitors conversations for escalation triggers.

    Analyzes conversation patterns to detect when human intervention is needed.
    """

    def __init__(self, system_prompt: str, client, node_schema=None, node_name="Escalation Monitor"):
        self.sys_prompt = system_prompt
        super().__init__(self.sys_prompt)

        self.client = client
        self.model_name = config.MODEL_ID_ESCALATION
        self.node_name = node_name
        self.schema = node_schema
        self.escalation_count = 0  # Track escalation triggers

    async def process_context(self, context: ConversationContext) -> AsyncGenerator[AgentResponse, None]:
        """
        Monitor conversation for escalation triggers.

        Args:
            context: Conversation context with messages.

        Yields:
            EscalationAlert: Escalation analysis results.
        """

        # Skip if escalation already detected
        if config.ESCALATION_DETECTED:
            logger.info("Escalation already in progress, skipping analysis")
            return

        if not context.events or len(context.events) < 3:  # Wait for some conversation
            logger.debug("Not enough conversation context for escalation analysis")
            return

        try:
            # Convert messages to OpenAI format
            openai_messages = convert_messages_to_openai(context.events, self.sys_prompt)

            # Prepare structured output format
            response_format = None
            if self.schema:
                response_format = {
                    "type": "json_schema",
                    "json_schema": {"name": "escalation_analysis", "strict": True, "schema": self.schema},
                }

            # Call OpenAI API for escalation analysis
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,
                max_tokens=100,
                temperature=config.TEMPERATURE,
                response_format=response_format,
            )

            extracted_info = response.choices[0].message.content

            if extracted_info:
                if response_format:
                    try:
                        # Parse structured escalation analysis
                        escalation_data = json.loads(extracted_info)
                        escalation_info = EscalationInfo.model_validate(escalation_data)

                        # Log escalation analysis
                        logger.info(f"🚨 {self.node_name} Analysis:")
                        logger.info(format_escalation_report(escalation_data))

                        # Check if escalation is needed
                        if escalation_info.escalation_level == "HIGH":
                            self.escalation_count += 1
                            logger.warning(f"HIGH escalation detected! Count: {self.escalation_count}")

                            # Yield escalation alert
                            yield config.EscalationAlert(
                                escalation_info=f"ESCALATION NEEDED: {escalation_info.reason}", urgency="high"
                            )

                        elif escalation_info.escalation_level == "MEDIUM":
                            logger.info(f"Medium escalation risk detected: {escalation_info.reason}")

                            yield config.EscalationAlert(
                                escalation_info=f"Escalation risk: {escalation_info.reason}", urgency="medium"
                            )

                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning(f"Failed to parse escalation analysis as JSON: {e}")
                        logger.info(f"Raw escalation feedback: {extracted_info}")

                        # Look for escalation keywords in unstructured response
                        escalation_keywords = [
                            "escalate",
                            "frustrated",
                            "angry",
                            "unresolved",
                            "human",
                            "manager",
                        ]
                        if any(keyword in extracted_info.lower() for keyword in escalation_keywords):
                            yield config.EscalationAlert(
                                escalation_info=f"Potential escalation detected: {extracted_info}",
                                urgency="medium",
                            )

                else:
                    # Unstructured analysis
                    logger.info(f"🚨 {self.node_name} unstructured feedback: {extracted_info}")

                    # Simple keyword-based escalation detection
                    escalation_keywords = [
                        "escalate",
                        "frustrated",
                        "angry",
                        "unresolved",
                        "human",
                        "manager",
                    ]
                    if any(keyword in extracted_info.lower() for keyword in escalation_keywords):
                        yield config.EscalationAlert(
                            escalation_info=f"Escalation indicators detected: {extracted_info}",
                            urgency="medium",
                        )

            else:
                logger.debug("No escalation analysis extracted")

        except Exception as e:
            logger.exception(f"Error during escalation analysis: {e}")
