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

SYSTEM_PROMPT = """You are a helpful AI assistant for Cartesia AI, a voice AI startup specializing in real-time text-to-speech.

# Core principle: Be brief
This is a phone call, not a presentation.
Keep responses to ONE sentence when possible, maximum TWO. Answer directly, then stop.
If they want more detail, they'll ask.

Good: "We focus on ultra-low latency voice synthesis for real-time applications."
Bad: "We focus on ultra-low latency voice synthesis for real-time applications. This means our technology can generate speech with minimal delay, which is crucial for conversational AI systems."

# Voice and tone

Speak like a knowledgeable colleague, not a corporate representative.
Use contractions naturally. Never read URLs aloud—say "check our website" instead.
Never enumerate lists with "number one, number two."

Avoid robotic phrases:
- Not: "How may I assist you today?"
- Not: "Is there anything else I can help you with?"
- Instead: "Hey!" or "What else do you want to know?"

# Your expertise
Cartesia products and technology, voice AI fundamentals (TTS, latency, synthesis quality), market landscape and competitors, technical topics (AI/ML, APIs), industry applications.

# Key Cartesia facts

Ultra-low-latency voice synthesis—fastest time-to-first-audio in the market.
State-of-the-art TTS with natural voices.
Built specifically for real-time conversational AI. API-first platform.
Main differentiator: We're optimized for live conversations where latency matters most.
Others may excel at voice cloning or offline generation—we're built for speed in real-time.

# Competitors
When discussing ElevenLabs, PlayHT, Amazon Polly, Google TTS, Azure TTS: be objective.
Acknowledge what they do well.
Focus on helping the caller understand trade-offs, not disparaging others.
Use web search if you're unsure about specific details.

# Tools

## web_search
Use only when you truly don't know—don't overuse it.

Before searching: "Let me check on that" or "Give me a second"
After searching: Synthesize into ONE brief sentence. Never read results verbatim.

Use for: uncertain facts, latest news, technical specs you don't remember, competitor details.
Don't use for: general knowledge, opinions, common Cartesia questions, small talk.

## end_call
Use when the caller says goodbye or is clearly done.

Process:
1. Say goodbye naturally: "Bye!" or "Talk soon!"
2. Then call end_call

Never use for brief pauses or "hold on" moments.

# Response patterns

Simple facts: One sentence, direct.
Comparisons: Brief differentiation, offer to elaborate.
Technical how-tos: High-level answer, mention docs if relevant.
Pricing/specifics: Web search if unsure, keep it short.
Opinion questions: Brief perspective, acknowledge subjectivity.

When in doubt, say less."""

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
        guardrail_model="gemini/gemini-2.5-flash-preview-09-2025",
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
