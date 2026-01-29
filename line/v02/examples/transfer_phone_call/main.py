"""
Transfer Phone Call Example - Navigate IVR menus and transfer calls.

Run with: ANTHROPIC_API_KEY=your-key uv run python main.py
"""

import os

from line.v02.llm_agent import LlmAgent, LlmConfig, end_call, send_dtmf, transfer_call
from line.v02.voice_agent_app import AgentEnv, CallRequest, VoiceAgentApp

SYSTEM_PROMPT = """You are a helpful phone assistant that can navigate automated phone systems \
and transfer calls.

You have two special capabilities:
1. **DTMF tones**: When you hear an automated menu asking to "press 1 for sales, press 2 for \
support", use the send_dtmf tool to press the appropriate button.
2. **Call transfer**: When the user wants to be connected to a specific phone number, use the \
transfer_call tool.

When navigating phone menus:
- Listen carefully to the menu options
- Ask the user which option they want if unclear
- Press the appropriate button using send_dtmf

When transferring calls:
- ALWAYS read back the full phone number and ask the user to confirm before transferring
- Only call the transfer_call tool AFTER the user confirms the number is correct
- Phone numbers must be in E.164 format (e.g., +14155551234)
- Example: "I have the number plus 1 4 1 5 5 5 5 1 2 3 4. Is that correct?"

Always be helpful and let the user know what you're doing."""

INTRODUCTION = (
    "Hello! I'm your phone assistant. I can help you navigate automated phone menus "
    "by pressing buttons, or transfer your call to another number. How can I help?"
)


async def get_agent(env: AgentEnv, call_request: CallRequest):
    return LlmAgent(
        model="anthropic/claude-sonnet-4-20250514",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        tools=[send_dtmf, transfer_call, end_call],
        config=LlmConfig(
            system_prompt=SYSTEM_PROMPT,
            introduction=INTRODUCTION,
        ),
    )


app = VoiceAgentApp(get_agent=get_agent)

if __name__ == "__main__":
    app.run()
