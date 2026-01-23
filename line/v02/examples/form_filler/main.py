"""
Form Filler Example - Collects user information via YAML-defined form.

Run with: GEMINI_API_KEY=your-key uv run python main.py
"""

import os
from pathlib import Path

from form_filler import FormFiller
from loguru import logger

from line.call_request import CallRequest
from line.v02.llm import LlmAgent, LlmConfig
from line.v02.tools import end_call
from line.v02.voice_agent_app import AgentEnv, VoiceAgentApp

FORM_PATH = Path(__file__).parent / "schedule_form.yaml"


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

    # Get the first question to include in the introduction
    first_question = form.get_current_question_text()

    return LlmAgent(
        model="gemini/gemini-2.0-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        tools=[form.tool, end_call],
        config=LlmConfig(
            system_prompt=form.get_system_prompt(),
            introduction=f"Hi! I'm here to collect some information from you. {first_question}",
        ),
    )


app = VoiceAgentApp(get_agent=get_agent)

if __name__ == "__main__":
    print("Starting Form Filler app...")
    app.run()
