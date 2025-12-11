"""
ConversationContext - Data structure for conversation state in ReasoningNode template method.

This class provides a clean abstraction for conversation data that gets passed
to specialized processing methods in ReasoningNode subclasses.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

from line.events import AgentResponse, AgentSpeechSent, EventInstance, UserTranscriptionReceived


@dataclass
class ConversationContext:
    """
    Encapsulates conversation state for ReasoningNode template method pattern.

    This standardizes how conversation data is passed between the template method
    (ReasoningNode.generate) and specialized processing (process_context).

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

    def get_committed_transcript(self) -> list[Union[UserTranscriptionReceived, AgentResponse]]:
        """Get all committed transcript messages from the conversation events."""
        committed_turns = []
        for event in self.events:
            if isinstance(event, UserTranscriptionReceived):
                committed_turns.append(event)
            elif isinstance(event, AgentSpeechSent):
                committed_turns.append(AgentResponse(content=event.content))
        return committed_turns
