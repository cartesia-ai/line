"""
Guardrails Wrapper Example - Cartesia AI Assistant with Content Filtering.

This example demonstrates how to wrap an LlmAgent with preprocessing and postprocessing
guardrails that:
- Detect and redact PII (emails, phone numbers, SSNs, credit cards)
- Block toxic/abusive content
- Detect prompt injection attempts
- Keep conversations on-topic (Cartesia, voice AI, software, AI landscape)
- End the call after repeated violations

The wrapper uses a separate LLM call (batched for efficiency) to classify user input
before passing it to the main agent.

Architecture:
- Inner agent: Anthropic Claude (supports web search + function calling together)
- Guardrail LLM: Gemini Flash (fast/cheap, used only for classification - no tools needed)

Run with: ANTHROPIC_API_KEY=your-key GEMINI_API_KEY=your-key uv run python main.py
"""

import os

from guardrails import GuardrailConfig, GuardrailsWrapper
from loguru import logger

from line.llm_agent import LlmAgent, LlmConfig, end_call, web_search
from line.voice_agent_app import AgentEnv, CallRequest, VoiceAgentApp

SYSTEM_PROMPT = """You are a helpful AI assistant for Cartesia AI, a voice AI startup that \
specializes in real-time text-to-speech and voice synthesis technology.

CRITICAL - BE BRIEF:
- Keep responses to ONE sentence when possible, two max
- Answer directly, then stop - don't over-explain
- Let the caller ask follow-ups if they want more detail

This is a voice conversation. Your responses will be spoken aloud:
- Speak naturally, like a real phone conversation
- NEVER read URLs, bullet points, or formatted text
- NEVER use robotic phrases like "How can I assist you?"
- When ending the call, say bye naturally then call end_call

Your role is to:
- Answer questions about Cartesia's products, technology, and capabilities
- Discuss voice AI, text-to-speech, speech synthesis, and related technologies
- Compare Cartesia with competitors like ElevenLabs, PlayHT, Amazon Polly, Google TTS, etc.
- Discuss the voice AI market, trends, and landscape
- Help with general AI/ML and software engineering questions

You have web search - use it when the caller asks about specific facts, current info, or things \
you're unsure about. Say "let me check on that" first, then give a brief answer. For general \
conversation or opinions, just respond directly without searching.

Key facts about Cartesia:
- Cartesia AI focuses on ultra-low-latency voice synthesis
- The company is building state-of-the-art TTS models
- Cartesia's technology is designed for real-time conversational AI applications

When discussing competitors, be objective and factual. Highlight Cartesia's strengths without \
disparaging others."""

INTRODUCTION = (
    "Hey there! Thanks for calling Cartesia. I can tell you about our voice technology, "
    "how we compare to other providers, or chat about the voice AI space in general. "
    "What can I help you with?"
)


async def get_agent(env: AgentEnv, call_request: CallRequest):
    """Create a Cartesia AI assistant wrapped with guardrails."""
    logger.info(f"Starting new call: {call_request.call_id}")

    # Create the inner LLM agent
    # Using Anthropic Claude which supports web search + function calling together
    # (Gemini's standard API doesn't support this combination)
    inner_agent = LlmAgent(
        model="anthropic/claude-sonnet-4-20250514",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        tools=[web_search, end_call],
        config=LlmConfig(
            system_prompt=SYSTEM_PROMPT,
            introduction=INTRODUCTION,
        ),
    )

    # Configure guardrails
    guardrail_config = GuardrailConfig(
        allowed_topics=(
            "Cartesia AI, voice AI, text-to-speech (TTS), speech synthesis, "
            "voice cloning, AI/ML, software engineering, competitors like ElevenLabs, "
            "PlayHT, Amazon Polly, Google Cloud TTS, Microsoft Azure Speech, "
            "and the voice AI market landscape"
        ),
        guardrail_model="gemini/gemini-2.0-flash",
        guardrail_api_key=os.getenv("GEMINI_API_KEY"),
        max_violations_before_end_call=3,
        # Custom messages
        toxic_response=(
            "I'd prefer to keep our conversation respectful. "
            "Is there something about Cartesia or voice AI I can help you with?"
        ),
        injection_response=(
            "I'm here specifically to help with questions about Cartesia and voice AI. "
            "What would you like to know about our technology?"
        ),
        off_topic_warning=(
            "I'm specifically here to help with questions about Cartesia, voice AI, "
            "and related topics. Is there something in that area I can help with?"
        ),
        end_call_message=(
            "It seems like you might have other things on your mind right now. "
            "Feel free to call back when you're ready to chat about Cartesia or voice AI. "
            "Have a great day!"
        ),
    )

    # Wrap the agent with guardrails
    return GuardrailsWrapper(inner_agent, guardrail_config)


app = VoiceAgentApp(get_agent=get_agent)

if __name__ == "__main__":
    print("Starting Cartesia AI Assistant with Guardrails...")
    print()
    print("Models:")
    print("  - Inner agent: Anthropic Claude (web search + tools)")
    print("  - Guardrail LLM: Gemini Flash (fast classification)")
    print()
    print("Guardrails enabled:")
    print("  - PII detection and redaction")
    print("  - Toxicity blocking")
    print("  - Prompt injection detection")
    print("  - Topic enforcement (Cartesia, voice AI, AI/ML)")
    print("  - Auto end-call after 3 violations")
    print()
    app.run()
