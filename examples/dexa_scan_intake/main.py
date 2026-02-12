"""DEXA Scan Intake Agent with knowledge base and Exa web search."""

import os
import time
from typing import AsyncIterable, List

from loguru import logger

from line.events import AgentSendText
from line.llm_agent import LlmAgent, LlmConfig, end_call
from line.voice_agent_app import AgentEnv, CallRequest, VoiceAgentApp

from tools import lookup_past_appointments, search_dexa_info, lookup_dexa_knowledge
from intake_form import (
    start_intake_form,
    record_intake_answer,
    get_intake_form_status,
    restart_intake_form,
    submit_intake_form,
    edit_intake_answer,
    go_back_in_intake_form,
    list_intake_answers,
    reset_form_instance,
)
from appointment_scheduler import (
    list_locations,
    check_availability,
    select_appointment_slot,
    book_appointment,
    send_availability_link,
    reset_scheduler_instance,
)


class TTFCTracker:
    """Tracks time to first token (TTFT) and time to first text chunk (TTFC) for LLM responses."""

    def __init__(self, log_interval: int = 5):
        self._ttft_times: List[float] = []  # Time to first token (any output)
        self._ttfc_times: List[float] = []  # Time to first text chunk
        self._turn_count: int = 0
        self._log_interval = log_interval

    def record_ttft(self, ttft_ms: float):
        """Record time to first token (any output from LLM)."""
        self._ttft_times.append(ttft_ms)
        logger.info(f"TTFT turn {self._turn_count + 1}: {ttft_ms:.1f}ms")

    def record_ttfc(self, ttfc_ms: float):
        """Record time to first text chunk."""
        self._ttfc_times.append(ttfc_ms)
        self._turn_count += 1
        logger.info(f"TTFC turn {self._turn_count}: {ttfc_ms:.1f}ms")

        if self._turn_count % self._log_interval == 0:
            ttft_avg = sum(self._ttft_times) / len(self._ttft_times) if self._ttft_times else 0
            ttfc_avg = sum(self._ttfc_times) / len(self._ttfc_times) if self._ttfc_times else 0
            logger.info(f"Averages over {self._turn_count} turns - TTFT: {ttft_avg:.1f}ms, TTFC: {ttfc_avg:.1f}ms")

    def reset(self):
        """Reset tracking for a new call."""
        self._ttft_times = []
        self._ttfc_times = []
        self._turn_count = 0


class TTFCWrappedAgent:
    """Wraps an LlmAgent to track time to first token and first text chunk."""

    def __init__(self, agent: LlmAgent, tracker: TTFCTracker):
        self._agent = agent
        self._tracker = tracker

    async def process(self, env, event) -> AsyncIterable:
        """Process an event and track TTFT and TTFC."""
        start_time = time.perf_counter()
        first_token_seen = False
        first_text_seen = False

        async for output in self._agent.process(env, event):
            if not first_token_seen:
                ttft_ms = (time.perf_counter() - start_time) * 1000
                self._tracker.record_ttft(ttft_ms)
                first_token_seen = True

            if not first_text_seen and isinstance(output, AgentSendText):
                ttfc_ms = (time.perf_counter() - start_time) * 1000
                self._tracker.record_ttfc(ttfc_ms)
                first_text_seen = True

            yield output


# Global TTFC tracker instance (reset per call)
_ttfc_tracker = TTFCTracker(log_interval=5)


SYSTEM_PROMPT = """You are a friendly voice assistant for a DEXA scanning facility. Keep responses SHORT - this is a phone call.

# Tools Available
- lookup_dexa_knowledge: Answer DEXA questions (topics: what_is_dexa, how_it_works, safety, preparation, etc.)
- search_dexa_info: Web search for current info (say "let me look that up" first)
- lookup_past_appointments: Find patient history (need: first_name, last_name, date_of_birth as YYYY-MM-DD)
- Intake form: start_intake_form, record_intake_answer, get_intake_form_status, edit_intake_answer, submit_intake_form
- Scheduling: list_locations, check_availability, select_appointment_slot, book_appointment

# Intake Form Rules
- Ask ONE question at a time
- For name/email/phone: FIRST say it back to confirm ("so that's john at gmail dot com?"), THEN call record_intake_answer with the value AFTER user confirms
- For all other fields (yes/no, numbers, etc): call record_intake_answer immediately with the answer
- NEVER call record_intake_answer with "yes" or "correct" - only call it with the actual answer value
- Field IDs for edits: first_name, last_name, email, phone, date_of_birth, ethnicity, gender, height_inches, weight_pounds

# Locations
5 SF locations: Financial District, SoMa, Marina, Castro, Sunset

# Voice Style
- 1-2 sentences max. Be brief.
- Use contractions: you'll, it's, we've
- Occasional fillers: "um", "so", "let's see"
- Never use bullet points or special characters
- Never start with "Great question!" - just answer directly
"""

INTRODUCTION = (
    "Hey! Thanks for calling. So, I can help you with pretty much anything about DEXA scans, "
    "What can I help you with?"
)


async def get_agent(env: AgentEnv, call_request: CallRequest):
    logger.info(f"Starting new DEXA intake call: {call_request}")

    # Reset state for new call
    reset_form_instance()
    reset_scheduler_instance()
    _ttfc_tracker.reset()

    agent = LlmAgent(
        model="gemini/gemini-2.5-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        tools=[
            lookup_dexa_knowledge,
            search_dexa_info,
            lookup_past_appointments,
            start_intake_form,
            record_intake_answer,
            get_intake_form_status,
            restart_intake_form,
            submit_intake_form,
            edit_intake_answer,
            go_back_in_intake_form,
            list_intake_answers,
            list_locations,
            check_availability,
            select_appointment_slot,
            book_appointment,
            send_availability_link,
            end_call,
        ],
        config=LlmConfig(
            system_prompt=SYSTEM_PROMPT,
            introduction=INTRODUCTION,
            temperature=1,
        ),
    )

    # Wrap agent with TTFC tracking
    return TTFCWrappedAgent(agent, _ttfc_tracker)


app = VoiceAgentApp(get_agent=get_agent)

if __name__ == "__main__":
    app.run()
