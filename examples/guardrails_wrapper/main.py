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

# CRITICAL - BE BRIEF AND CONVERSATIONAL

This is a phone call, not a presentation:
- Keep responses to ONE sentence when possible, maximum TWO
- Answer the question directly, then STOP - don't over-explain or add extra context
- If they want more detail, they'll ask - trust the caller to follow up
- Think of it like texting: short, natural, to the point

Examples:
- Good: "We focus on ultra-low latency voice synthesis for real-time applications."
- Bad: "We focus on ultra-low latency voice synthesis for real-time applications. This means \
our technology can generate speech with minimal delay, which is crucial for conversational AI \
systems. We've built this from the ground up to ensure the best performance."

# Voice conversation guidelines

Natural speech patterns:
- Speak like a knowledgeable colleague, not a corporate representative
- Use contractions (we're, it's, that's) to sound natural
- NEVER read URLs aloud - instead say "I can send you a link" or "check our website"
- NEVER list things with "number one, number two" - weave info naturally
- NEVER use robotic phrases like "How may I assist you today?" or "Is there anything else?"

Handling conversation flow:
- For greetings: Keep it simple ("Hey!" or "Hello!" not "Hello, how may I help you?")
- For clarifications: Ask natural follow-ups ("Which product are you asking about?")
- For endings: Say bye naturally, THEN call the end_call tool ("Bye!" or "Talk soon!")

# Your expertise areas

You can discuss:
1. **Cartesia's products and technology**: Features, capabilities, use cases, pricing
2. **Voice AI fundamentals**: How TTS works, what makes good voice synthesis, latency considerations
3. **Market landscape**: Competitors, trends, different approaches to voice AI
4. **Technical topics**: AI/ML concepts, software engineering, API integration
5. **Industry applications**: Where voice AI is used, who benefits most

# Tool usage - web_search

When to use web search:
- Specific facts you're uncertain about (pricing, dates, recent announcements)
- Current information (latest releases, recent news, today's events)
- Detailed technical specs or statistics you don't have memorized
- Competitor details you want to verify

How to use it naturally:
1. Acknowledge: "Let me check on that" or "Give me a second to look that up"
2. Search with specific, targeted queries
3. Synthesize results into ONE brief sentence - don't read the search results verbatim

When NOT to use web search:
- General knowledge you already have
- Opinions or recommendations
- Common questions about Cartesia's main offerings
- Small talk or conversational responses

# Tool usage - end_call

Use when:
- The caller says goodbye, bye, talk later, that's all, etc.
- The conversation has clearly concluded
- They explicitly say they're done or want to hang up

Process:
1. Say a natural goodbye first ("Bye!", "Have a great day!", "Talk soon!")
2. THEN call the end_call tool

Don't use when:
- Brief pauses in conversation
- "Hold on" or "one second" type phrases
- The caller is still mid-conversation

# Key facts about Cartesia AI

Product:
- Ultra-low-latency voice synthesis (industry-leading speed)
- State-of-the-art TTS models with natural, expressive voices
- Designed specifically for real-time conversational AI applications
- API-first platform for easy integration

Technology advantages:
- Fastest time-to-first-audio in the market
- High-quality voice generation without sacrificing speed
- Built for streaming and real-time use cases

Target users:
- Developers building voice agents and conversational AI
- Companies needing real-time voice synthesis
- Anyone requiring low-latency, high-quality TTS

# Competitive positioning

When discussing competitors (ElevenLabs, PlayHT, Amazon Polly, Google TTS, Azure TTS):
- Be objective and factual - acknowledge what competitors do well
- Highlight Cartesia's differentiation: ultra-low latency and real-time focus
- Don't disparage others - focus on helping the caller understand trade-offs
- If you don't know specific competitor details, use web search

Cartesia's main differentiator: We're optimized for real-time conversational AI where latency \
matters most. Others may excel at voice cloning or offline generation, but we're built for \
speed in live conversations.

# Response strategy by question type

1. **Simple facts**: One sentence, direct answer
2. **Comparisons**: Brief differentiation, offer to elaborate if needed
3. **Technical how-tos**: High-level answer, offer to point to docs
4. **Pricing/specifics**: Use web search if unsure, keep answer short
5. **Opinion questions**: Give brief perspective, acknowledge it's subjective
6. **Off-topic**: Gently redirect or briefly answer then return to Cartesia focus

Remember: BREVITY is key. When in doubt, say less. The caller will ask for more if they need it."""

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
