"""
Example agent demonstrating the inactivity timeout feature.

When the user doesn't start speaking within a configured timeout after the agent
finishes, an InactivityTimeout event is fired. This agent handles that event by
re-prompting the user.

Usage:
    ANTHROPIC_API_KEY=your-key uv run python main.py
"""

import os

from loguru import logger

from line.events import InactivityTimeout
from line.llm_agent import LlmAgent, LlmConfig
from line.voice_agent_app import AgentEnv, CallRequest, VoiceAgentApp

# Inactivity timeout in milliseconds (5 seconds)
INACTIVITY_TIMEOUT_MS = 5000

SYSTEM_PROMPT = """You are a friendly voice assistant designed for natural conversation.

# Personality
Warm, patient, and helpful. You understand that users may sometimes need a moment to respond.

# Voice and tone
Speak naturally and conversationally. Keep responses brief (1-2 sentences).

# Handling silence
If the user hasn't responded (indicated by an inactivity timeout), gently re-engage them with
a friendly prompt. Vary your re-engagement phrases:
- "Are you still there?"
- "I'm here whenever you're ready."
- "Take your time - just let me know when you'd like to continue."
- "Is there anything else I can help you with?"

Don't be pushy or repeat the same phrase. If the user remains silent after multiple prompts,
gracefully offer to end the call."""

INTRODUCTION = "Hi there! I'm your voice assistant. What can I help you with today?"


class InactivityAwareAgent:
    """Wrapper agent that handles InactivityTimeout events with custom behavior."""

    def __init__(self, llm_agent: LlmAgent):
        self.llm_agent = llm_agent
        self.inactivity_count = 0
        self.max_inactivity_prompts = 3

    async def process(self, env, event):
        # Track inactivity events
        if isinstance(event, InactivityTimeout):
            self.inactivity_count += 1
            logger.info(f"Inactivity timeout #{self.inactivity_count} (timeout_ms={event.timeout_ms})")

            # After max prompts, the LLM will handle graceful ending via system prompt
            if self.inactivity_count > self.max_inactivity_prompts:
                logger.info("Max inactivity prompts reached")
        else:
            # Reset counter on any user activity
            self.inactivity_count = 0

        # Delegate to the LLM agent for response generation
        async for output in self.llm_agent.process(env, event):
            yield output


async def get_agent(env: AgentEnv, call_request: CallRequest):
    logger.info(f"Starting new call: {call_request.call_id}")

    # Set default inactivity timeout if not configured by caller
    # This is picked up by ConversationRunner which is created after get_agent returns
    if call_request.agent.inactivity_timeout_ms is None:
        call_request.agent.inactivity_timeout_ms = INACTIVITY_TIMEOUT_MS

    logger.info(f"Inactivity timeout: {call_request.agent.inactivity_timeout_ms}ms")

    llm_agent = LlmAgent(
        model="anthropic/claude-haiku-4-5-20251001",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        config=LlmConfig.from_call_request(
            call_request,
            fallback_system_prompt=SYSTEM_PROMPT,
            fallback_introduction=INTRODUCTION,
        ),
    )

    return InactivityAwareAgent(llm_agent)


app = VoiceAgentApp(get_agent=get_agent)

if __name__ == "__main__":
    print(f"Starting inactivity timeout example (timeout={INACTIVITY_TIMEOUT_MS}ms)")
    app.run()
