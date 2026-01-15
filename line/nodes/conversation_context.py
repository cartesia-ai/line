"""
ConversationContext - Data structure for conversation state in ReasoningNode template method.

This class provides a clean abstraction for conversation data that gets passed
to specialized processing methods in ReasoningNode subclasses.
"""

from dataclasses import dataclass, field
import re
from typing import Any, List, Optional

from line.events import (
    AgentResponse,
    AgentSpeechSent,
    EventInstance,
    UserTranscriptionReceived,
)

NORMAL_CHARACTERS_REGEX = r"(\s+|[^\w\s]+)"


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
                if committed_text:  # Only add if there's actual content
                    committed_events.append(AgentResponse(content=committed_text))
            # All other events are committed as is
            else:
                committed_events.append(event)

        return committed_events

    def _parse_committed(self, pending_text: str, speech_text: str) -> tuple[str, list[str]]:
        pending_parts = list(filter(lambda x: x != "", re.split(NORMAL_CHARACTERS_REGEX, pending_text)))
        speech_parts = list(filter(lambda x: x != "", re.split(NORMAL_CHARACTERS_REGEX, speech_text)))

        # If the pending text has no spaces (ex. non-latin languages), commit the entire pending text.
        if len([x for x in pending_parts if x.isspace()]) == 0:
            return speech_text, ""

        committed_parts = []
        still_pending_text = []
        for pending_part in pending_parts:
            # If speech_text is empty), treat remaining pending parts as still pending.
            if not speech_parts:
                still_pending_text.extend(pending_part)
            # If the next pending text matches the start of what's been marked committed (as sent by TTS),
            # add it to committed and trim it from speech_parts.
            elif speech_parts[0].startswith(pending_part):
                speech_parts[0] = speech_parts[0][len(pending_part) :]
                committed_parts.extend(pending_part)
                if len(speech_parts[0]) == 0:
                    speech_parts.pop(0)
            # If the part is purely whitespace, add it directly to committed_parts.
            elif pending_part.isspace():
                committed_parts.extend(pending_part)
            # Otherwise, this part isn't aligned with the committed speech
            # (possibly an interruption or TTS mismatch);
            else:
                pass

        committed_str = "".join(committed_parts).strip()
        return committed_str, "".join(still_pending_text)
