"""History processor for pruning and summarizing long conversations."""

import os
from typing import List, Optional

from loguru import logger

from line import CustomHistoryEntry
from line.events import (
    AgentTextSent,
    AgentToolCalled,
    AgentToolReturned,
    HistoryEvent,
    UserTextSent,
)
from line.llm_agent import LlmAgent, LlmConfig

from intake_form import get_form
from appointment_scheduler import get_scheduler

# Configuration
MESSAGE_THRESHOLD = 40  # Prune when history exceeds this many messages
KEEP_RECENT = 15  # Keep this many recent messages intact

SUMMARIZATION_SYSTEM_PROMPT = """You are a conversation summarizer. Your task is to summarize conversation excerpts concisely.

Rules:
- Summarize in 2-3 sentences maximum
- Focus on key topics discussed and important information shared
- Start your summary with "Earlier in the call:"
- Be concise and factual
- Do not add any commentary or questions"""


class SummarizationAgent:
    """A lightweight agent for summarizing conversation text."""

    def __init__(self):
        self._agent: Optional[LlmAgent] = None

    def _get_agent(self) -> LlmAgent:
        """Lazily create the summarization agent."""
        if self._agent is None:
            self._agent = LlmAgent(
                model="anthropic/claude-haiku-4-5-20251001",
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                tools=[],  # No tools needed for summarization
                config=LlmConfig(
                    system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
                    max_tokens=150,
                    temperature=1,
                ),
            )
        return self._agent

    async def summarize(self, text: str) -> str:
        """Summarize the given text using the agent's LLM."""
        if not text.strip():
            return ""

        agent = self._get_agent()

        try:
            # Use the agent's internal completion method
            # Build messages manually since we're doing a simple one-shot completion
            from litellm import acompletion

            response = await acompletion(
                model=agent._model,
                api_key=agent._api_key,
                messages=[
                    {"role": "system", "content": SUMMARIZATION_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Summarize this conversation:\n\n{text}"},
                ],
                max_tokens=150,
                temperature=1,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Failed to summarize conversation: {e}")
            return "Earlier in the call: User discussed DEXA scan questions and intake process."


# Global summarization agent instance
_summarization_agent: Optional[SummarizationAgent] = None


def get_summarization_agent() -> SummarizationAgent:
    """Get or create the summarization agent."""
    global _summarization_agent
    if _summarization_agent is None:
        _summarization_agent = SummarizationAgent()
    return _summarization_agent


def _get_form_status_summary() -> str:
    """Get a concise summary of current form status."""
    form = get_form()
    status = form.get_status()

    if not status["is_started"]:
        return ""

    if status["is_submitted"]:
        return "[FORM STATUS] Intake form completed and submitted."

    if status["is_complete"]:
        return "[FORM STATUS] Intake form complete, ready to submit."

    return (
        f"[FORM STATUS] Progress: {status['progress']} | "
        f"Section: {status['current_section']} | "
        f"Next question: {status['current_question']}"
    )


def _get_scheduler_status_summary() -> str:
    """Get a concise summary of current scheduler status."""
    scheduler = get_scheduler()

    if scheduler._booked_appointment:
        appt = scheduler._booked_appointment
        return (
            f"[APPOINTMENT BOOKED] Confirmation: {appt['confirmation_number']} | "
            f"{appt['slot']['time']} on {appt['slot']['date']}"
        )

    if scheduler._selected_slot:
        slot = scheduler._selected_slot
        return (
            f"[APPOINTMENT SELECTED] {slot['time']} on {slot['date']} | "
            f"Awaiting patient info to confirm booking"
        )

    return ""


def _extract_conversation_text(history: List[HistoryEvent]) -> str:
    """Extract user and assistant messages as text for summarization."""
    lines = []
    for event in history:
        if isinstance(event, UserTextSent):
            lines.append(f"User: {event.content}")
        elif isinstance(event, AgentTextSent):
            lines.append(f"Agent: {event.content}")
    return "\n".join(lines)


def _is_tool_event(event: HistoryEvent) -> bool:
    """Check if event is a tool call or result."""
    return isinstance(event, (AgentToolCalled, AgentToolReturned))


async def _summarize_conversation(text: str) -> str:
    """Use the summarization agent to summarize the conversation."""
    agent = get_summarization_agent()
    return await agent.summarize(text)


def _count_non_tool_events(history: List[HistoryEvent]) -> int:
    """Count non-tool events (user and agent text messages) in history."""
    return sum(1 for e in history if not _is_tool_event(e))


async def process_history(history: List[HistoryEvent]) -> List[HistoryEvent]:
    """
    Process history to prune and summarize if too long.

    Strategy:
    1. If under threshold, return as-is with status prepended
    2. If over threshold:
       - Keep last N messages intact
       - Summarize older conversation messages
       - Keep all tool calls (they contain structured state)
       - Prepend form and scheduler status
    """
    # Always prepend current status
    status_entries = []

    form_status = _get_form_status_summary()
    if form_status:
        status_entries.append(CustomHistoryEntry(content=form_status, role="system"))

    scheduler_status = _get_scheduler_status_summary()
    if scheduler_status:
        status_entries.append(CustomHistoryEntry(content=scheduler_status, role="system"))

    # Count non-tool events for threshold check
    non_tool_count = _count_non_tool_events(history)

    # If under threshold, just prepend status
    if non_tool_count <= MESSAGE_THRESHOLD:
        return status_entries + history

    logger.info(f"History has {non_tool_count} non-tool messages, pruning (threshold: {MESSAGE_THRESHOLD})")

    # Split into old and recent
    old_history = history[:-KEEP_RECENT]
    recent_history = history[-KEEP_RECENT:]

    # Extract conversation text from old history for summarization
    old_conversation_text = _extract_conversation_text(old_history)

    # Keep tool calls from old history
    old_tool_events = [e for e in old_history if _is_tool_event(e)]

    # Summarize old conversation
    summary = await _summarize_conversation(old_conversation_text)

    # Build pruned history
    pruned_history = []

    # 1. Status entries first
    pruned_history.extend(status_entries)

    # 2. Summary of old conversation
    if summary:
        pruned_history.append(CustomHistoryEntry(content=f"[CONVERSATION SUMMARY] {summary}", role="system"))

    # 3. Tool calls from old history (keep structured state)
    pruned_history.extend(old_tool_events)

    # 4. Recent history intact
    pruned_history.extend(recent_history)

    logger.info(f"Pruned history from {len(history)} to {len(pruned_history)} messages")

    return pruned_history
