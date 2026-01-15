"""
ConversationContext - Data structure for conversation state in ReasoningNode template method.

This class provides a clean abstraction for conversation data that gets passed
to specialized processing methods in ReasoningNode subclasses.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional
import re
from loguru import logger

from line.events import (
    AgentResponse,
    AgentSpeechSent,
    EventInstance,
    UserTranscriptionReceived,
)


@dataclass
class ConversationContext:
    """
    Encapsulates conversation state for ReasoningNode template method pattern.

    Attributes:
        events: List of conversation events
        system_prompt: The system prompt for this reasoning node
        metadata: Additional context data for specialized processing
    """

    events: List[EventInstance]
    system_prompt: str
    metadata: dict = field(default_factory=dict)

    def format_events(self, max_messages: int = None) -> str:
        """
        Format conversation messages as a string for LLM prompts.

        Args:
            max_messages: Maximum number of recent messages to include

        Returns:
            Formatted conversation string
        """
        events = self.events
        if max_messages is not None:
            events = events[-max_messages:]

        return "\n".join(f"{type(event)}: {event}" for event in events)

    def get_latest_user_transcript_message(self) -> Optional[str]:
        """Get the most recent user message content."""
        for msg in reversed(self.events):
            if isinstance(msg, UserTranscriptionReceived):
                return msg.content
        return None

    def get_event_count(self) -> int:
        """Get total number of messages in context."""
        return len(self.events)

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata for specialized processing."""
        self.metadata[key] = value

    def get_committed_events(self) -> list[EventInstance]:
        logger.info(f"🧠 Getting committed events: {self.events}")
        pending_text = ""
        committed_events = []
        for event in self.events:
            if isinstance(event, AgentResponse):
                pending_text = pending_text + event.content
            elif isinstance(event, AgentSpeechSent):
                committed_text, pending_text = self._parse_committed(
                    pending_text,
                    event.content,
                )
                if committed_text.strip():  # Only add if there's actual content
                    committed_events.append(AgentResponse(content=committed_text))
            # All other events are committed as is
            else:
                committed_events.append(event)

        return committed_events
        
    def _parse_committed(self, pending_text: str, committed_text: str) -> tuple[str, list[str]]:
        pending_parts = re.split(r"(\s+)", pending_text)
        committed_parts = []
        still_pending_text = []
        for pending_part in pending_parts:
            if not committed_text:
                still_pending_text.extend(pending_part)
            elif pending_part.isspace():
                committed_parts.extend(pending_part)
            elif committed_text.startswith(pending_part):
                committed_text = committed_text[len(pending_part):]
                committed_parts.extend(pending_part)
            else:
                # skipped text
                pass


        return "".join(committed_parts), "".join(still_pending_text)