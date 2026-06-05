"""Outbound voice agent with voicemail detection.

Demonstrates both evaluation approaches on a real agent. For outbound calls the
agent should let the callee speak first, so we set ``introduction=""`` — the
agent stays silent until it hears the opening line (a live "Hello?" or a
recorded voicemail greeting), then decides how to proceed.

Select the approach with the ``VOICEMAIL_APPROACH`` env var:

    # Approach 1 — built-in voicemail tool (default). The main LM calls the
    # tool when it hears a machine greeting; the tool is auto-removed after the
    # first turn so it can't fire mid-conversation.
    ANTHROPIC_API_KEY=... VOICEMAIL_APPROACH=tool uv run python main.py

    # Approach 2 — cheap-LM detection sidecar. A separate classifier gates the
    # main LM's first reply and ends the call on a voicemail verdict.
    ANTHROPIC_API_KEY=... OPENAI_API_KEY=... VOICEMAIL_APPROACH=sidecar uv run python main.py
"""

import os

from loguru import logger

from line.llm_agent import LlmAgent, LlmConfig, VoicemailDetectionConfig, end_call, voicemail
from line.voice_agent_app import AgentEnv, CallRequest, VoiceAgentApp

SYSTEM_PROMPT = """You are an outbound voice agent making a brief courtesy call about the customer's recent order.

Wait for the other party to speak first. If you're talking to a real person, greet them warmly, confirm
you've reached the right customer, and keep it short. Speak in natural, conversational prose."""

# Outbound: stay silent until the callee speaks so we can hear the opening line.
INTRODUCTION = ""

VOICEMAIL_MESSAGE = "Hi, this is a quick courtesy call about your recent order. Please give us a call back when you can. Thank you!"


async def get_agent(env: AgentEnv, call_request: CallRequest):
    approach = os.getenv("VOICEMAIL_APPROACH", "tool").lower()
    logger.info(f"Starting outbound call {call_request.call_id} with voicemail approach={approach!r}")

    config = LlmConfig.from_call_request(
        call_request, fallback_system_prompt=SYSTEM_PROMPT, fallback_introduction=INTRODUCTION
    )

    if approach == "sidecar":
        # Approach 2: cheap-LM detection sidecar runs alongside the main LM.
        return LlmAgent(
            model="anthropic/claude-haiku-4-5-20251001",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            tools=[end_call],
            config=config,
            voicemail_detection=VoicemailDetectionConfig(
                model="openai/gpt-4o-mini",
                api_key=os.getenv("OPENAI_API_KEY"),
                message=VOICEMAIL_MESSAGE,
                initial_gate_ms=200,
            ),
        )

    # Approach 1: built-in voicemail tool, dropped after the first user turn.
    return LlmAgent(
        model="anthropic/claude-haiku-4-5-20251001",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        tools=[voicemail(message=VOICEMAIL_MESSAGE), end_call],
        config=config,
        voicemail_tool_active_turns=1,
    )


app = VoiceAgentApp(get_agent=get_agent)

if __name__ == "__main__":
    print("Starting app")
    app.run()
