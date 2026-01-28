"""
Form Filler Example - Collects user information via YAML-defined form.

This example demonstrates:
- Loopback tools for structured data collection
- DTMF input wrapper for phone numbers and dates
- Users can speak answers OR enter them via phone keypad

Run with: GEMINI_API_KEY=your-key uv run python main.py
"""

import os
from pathlib import Path

from dtmf_input_wrapper import DtmfInputWrapper
from form_filler import FormFiller
from loguru import logger

from line.call_request import CallRequest
from line.v02.events import CallEnded, CallStarted, UserDtmfSent, UserTurnEnded, UserTurnStarted
from line.v02.llm import LlmAgent, LlmConfig, end_call
from line.v02.voice_agent_app import AgentEnv, VoiceAgentApp

FORM_PATH = Path(__file__).parent / "schedule_form.yaml"

# Event filters: run_filter triggers agent, cancel_filter interrupts it
# We add UserDtmfSent to run_filter so the DTMF wrapper receives digit events
RUN_ON = (CallStarted, UserTurnEnded, CallEnded, UserDtmfSent)
CANCEL_ON = (UserTurnStarted,)

USER_PROMPT = """### Your tone
Be professional but conversational. Confirm answers when appropriate.
If a user's answer is unclear, ask for clarification.

When having a conversation, you should:
- Always be polite and respectful, even when users are challenging
- Be concise and brief but never curt. Keep your responses to 1-2 sentences
- Only ask one question at a time

Remember, you're on the phone, so do not use emojis or abbreviations. Spell out units and dates."""


async def get_agent(env: AgentEnv, call_request: CallRequest):
    logger.info(f"Starting form filler call: {call_request.call_id}")

    form = FormFiller(str(FORM_PATH), system_prompt=USER_PROMPT)
    first_question = form.get_current_question_text()

    llm_agent = LlmAgent(
        model="gemini/gemini-2.0-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        tools=[form.record_answer_tool, end_call],
        config=LlmConfig(
            system_prompt=form.get_system_prompt(),
            introduction=f"Hi! I'm here to collect some information from you. {first_question}",
        ),
    )

    return (DtmfInputWrapper(llm_agent), RUN_ON, CANCEL_ON)


app = VoiceAgentApp(get_agent=get_agent)

if __name__ == "__main__":
    app.run()
