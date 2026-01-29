"""Voice Agent with Real-time Web Form Filling using Browserbase.

This example demonstrates a voice agent that conducts phone questionnaires
while automatically filling out web forms in real-time using Stagehand
browser automation powered by Browserbase.

The conversation flow is deterministic:
1. Agent greets and asks if user is ready
2. User confirms -> start_questionnaire tool asks first question
3. User answers -> record_form_field tool records and asks next question
4. Repeat until all questions answered
5. Form is submitted and call ends
6. On CallEnded, cleanup browser resources

Required environment variables:
- GEMINI_API_KEY: Your Gemini API key
- BROWSERBASE_API_KEY: Your Browserbase API key
- BROWSERBASE_PROJECT_ID: Your Browserbase project ID
"""

import os
from typing import AsyncIterable

from loguru import logger
from stagehand_form_filler import StagehandFormFiller

from line.v02.agent import TurnEnv
from line.v02.events import CallEnded, InputEvent, OutputEvent
from line.v02.llm_agent import LlmAgent, LlmConfig
from line.v02.voice_agent_app import AgentEnv, CallRequest, VoiceAgentApp

# Model configuration
MODEL_ID = os.getenv("MODEL_ID", "gemini/gemini-2.0-flash")
# Target form URL - the actual web form to fill
FORM_URL = "https://forms.fillout.com/t/rff6XZTSApus"

SYSTEM_PROMPT = """
You are a friendly assistant helping users fill out a job application form.

### Your tools
- start_questionnaire: Call this when the user says they are ready to begin
- record_form_field: Call this after the user answers each question with the field name and their answer

### Instructions
1. When the user says they're ready (e.g., "yes", "sure", "let's go", "ready"), call start_questionnaire
2. After each user response, identify which field they answered and call record_form_field with:
   - field_name: The field being answered (full_name, email, phone, work_eligibility,
     availability_type, role_selection, previous_experience, skills_experience, additional_info)
   - value: The user's answer

### Important
- The tools handle asking the next question automatically - do not generate additional responses
- Just call the appropriate tool after each user input
- Listen carefully to extract the correct value from the user's response
"""


class FormFillingAgent:
    """Wrapper agent that handles CallEnded for cleanup.

    This agent delegates all events to the underlying LlmAgent, but intercepts
    CallEnded to perform cleanup of browser resources.
    """

    def __init__(self, form_filler: StagehandFormFiller):
        self.form_filler = form_filler
        self.llm_agent = LlmAgent(
            model=MODEL_ID,
            api_key=os.getenv("GEMINI_API_KEY"),
            tools=[
                form_filler.start_questionnaire,  # Called when user is ready to begin
                form_filler.record_form_field,  # Records answers and asks next question
            ],
            config=LlmConfig(
                system_prompt=SYSTEM_PROMPT,
                # Introduction matches v01: greeting + ask if ready
                introduction=(
                    "Hello! I'm here to help you fill out an application form today. "
                    "I'll ask you a series of questions and fill in the form as we go. "
                    "Ready to get started?"
                ),
            ),
        )

    async def process(self, env: TurnEnv, event: InputEvent) -> AsyncIterable[OutputEvent]:
        """Process an input event.

        Delegates to LlmAgent for most events, but handles CallEnded specially
        to ensure browser resources are cleaned up.
        """
        if isinstance(event, CallEnded):
            logger.info("Call ended - cleaning up browser resources")
            await self.form_filler.on_call_ended()
        else:
            async for output in self.llm_agent.process(env, event):
                yield output


async def get_agent(env: AgentEnv, call_request: CallRequest):
    """Create and configure the form-filling voice agent.

    This agent uses a wrapper around LlmAgent that handles CallEnded
    for cleanup. The passthrough tools provide deterministic conversation flow.

    Args:
        env: The agent environment.
        call_request: The incoming call request.

    Returns:
        A FormFillingAgent wrapping LlmAgent.
    """
    form_filler = StagehandFormFiller(form_url=FORM_URL)
    return FormFillingAgent(form_filler)


app = VoiceAgentApp(get_agent=get_agent)
if __name__ == "__main__":
    print("Starting Voice Agent with Web Form Automation")
    print(f"Will fill form at: {FORM_URL}")
    print("Ready to receive calls...")
    print("Form filling happens invisibly while processing voice calls.\n")
    app.run()
