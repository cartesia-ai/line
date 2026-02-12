"""Flight Change Agent — collects caller details and processes flight changes."""

import os
import time
from typing import AsyncIterable, List

from loguru import logger

from line.events import AgentSendText
from line.llm_agent import LlmAgent, LlmConfig
from line.voice_agent_app import AgentEnv, CallRequest, VoiceAgentApp

from booking_form import (
    start_booking_form,
    record_booking_answer,
    get_booking_form_status,
    change_flight,
    reset_form_instance,
)


class TTFCTracker:
    """Tracks time to first token (TTFT) and time to first text chunk (TTFC) for LLM responses."""

    def __init__(self, log_interval: int = 5):
        self._ttft_times: List[float] = []
        self._ttfc_times: List[float] = []
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


SYSTEM_PROMPT = """You are a friendly voice assistant for a flight change service. Keep responses SHORT - this is a phone call.

# Tools Available
- start_booking_form: Begin collecting caller details
- record_booking_answer: Record the caller's answer to the current question
- get_booking_form_status: Check form progress
- change_flight: Process the flight change after all 5 questions are answered

# Form Rules
- Ask ONE question at a time
- Record every answer immediately with record_booking_answer — no confirmation needed
- NEVER call record_booking_answer with "yes" or "correct" — only call it with the actual answer value

# Flow
1. The introduction already asks for the caller's first name. When they respond, call start_booking_form and then immediately record_booking_answer with their first name.
2. Continue asking the remaining 4 questions one at a time (last name, new date, time preference, confirmation code)
3. After all 5 answers, say "let me make that change for you" and then call change_flight
4. When change_flight returns, tell the caller it's all done
5. After helping, politely ask "Is there anything else I can help you with?"

# Voice Style
- 1-2 sentences max. Be brief.
- Use contractions: you'll, it's, we've
- Natural speech, like a real phone agent
- Never use bullet points or special characters
- Never start with "Great question!" — just answer directly
"""

INTRODUCTION = (
    "Hi! Thanks for calling. I'm happy to assist you with a flight change. "
    "Can I get your first name?"
)


async def get_agent(env: AgentEnv, call_request: CallRequest):
    logger.info(f"Starting new flight change call: {call_request}")

    # Reset state for new call
    reset_form_instance()
    _ttfc_tracker.reset()

    agent = LlmAgent(
        model="gemini/gemini-2.5-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        tools=[
            start_booking_form,
            record_booking_answer,
            get_booking_form_status,
            change_flight,
        ],
        config=LlmConfig(
            system_prompt=SYSTEM_PROMPT,
            introduction=INTRODUCTION,
            temperature=1,
        ),
    )

    return TTFCWrappedAgent(agent, _ttfc_tracker)


app = VoiceAgentApp(get_agent=get_agent)

if __name__ == "__main__":
    app.run()
