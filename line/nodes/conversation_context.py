"""
ConversationContext - Data structure for conversation state in ReasoningNode template method.

This class provides a clean abstraction for conversation data that gets passed
to specialized processing methods in ReasoningNode subclasses.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

from line.events import AgentResponse, AgentSpeechSent, EventInstance, UserTranscriptionReceived, ToolCall, ToolResult


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

    def get_committed_turns(self) -> list[Union[UserTranscriptionReceived, AgentResponse]]:
        """
        Get all transcript turns that were actually spoken by the agent.
        
        Matches AgentResponse events (with formatting) against AgentSpeechSent events
        (without spaces) to preserve original formatting for LLM context.
        
        Returns:
            List of committed turns: UserTranscriptionReceived and AgentResponse events
            that were confirmed spoken via AgentSpeechSent.
        """
        pending_responses = []
        committed_turns = []
        
        for event in self.events:
            if isinstance(event, UserTranscriptionReceived):
                committed_turns.append(event)
            elif isinstance(event, AgentResponse):
                pending_responses.append(event)
            elif isinstance(event, AgentSpeechSent):
                committed = self._process_speech_event(
                    event.content, 
                    pending_responses,
                )
                committed_turns.extend(committed)
        
        return committed_turns

    @staticmethod
    def _match_formatted_text_to_speech(formatted_text: str, speech_no_whitespace: str) -> tuple[str, int, int]:
        """
        Match AgentResponse text against whitespace-free AgentSpeechSent text.
        
        Performs character-by-character matching while preserving whitespace from AgentResponse text. Stops at first mismatch.
            
        Args:
            formatted_text: AgentResponse content with spaces/newlines (e.g., "Hello world!\n")
            speech_no_whitespace: AgentSpeechSent content without whitespace (e.g., "Helloworld!")
            
        Returns:
            (matched_content_with_formatting, chars_consumed_from_response, chars_consumed_from_speech)
        """
        matched_chars = []
        response_idx = 0
        speech_idx = 0
        
        while speech_idx < len(speech_no_whitespace) and response_idx < len(formatted_text):
            char = formatted_text[response_idx]
            if char.isspace():
                # Preserve all whitespace (spaces, newlines, tabs) from original formatting
                matched_chars.append(char)
                response_idx += 1
            elif char == speech_no_whitespace[speech_idx]:
                matched_chars.append(char)
                response_idx += 1
                speech_idx += 1
            else:
                break
        
        return "".join(matched_chars), response_idx, speech_idx
    
    def _process_speech_event(
        self, 
        speech_content: str, 
        pending_responses: list[AgentResponse],
    ) -> list[AgentResponse]:
        """
        Match AgentSpeechSent content against pending AgentResponse events.
        
        Args:
            speech_content: Content from AgentSpeechSent (no whitespace from word-level TTS)
            pending_responses: Pending AgentResponse events (mutated: matched items popped)
        
        Returns:
            List of committed AgentResponse events with preserved formatting for LLM context.
        """    
        committed = []
        remaining_speech = speech_content
        
        while pending_responses and remaining_speech:
            pending = pending_responses[0]
            # Remove all whitespace from pending response to match against speech (which has no spaces)
            normalized_pending = "".join(pending.content.split())
            
            if remaining_speech.startswith(normalized_pending):
                # Full match - commit the entire pending response
                committed.append(pending_responses.pop(0))
                remaining_speech = remaining_speech[len(normalized_pending):]
            else:
                # Partial match - find how much matches
                matched_content, chars_consumed_response, chars_consumed_speech = self._match_formatted_text_to_speech(
                    pending.content, 
                    remaining_speech
                )
                
                # Part of the response was spoken and committed
                # Only commit if there's actual content (not just whitespace or empty string)
                if matched_content.strip():
                    committed.append(AgentResponse(content=matched_content))
                    remaining_content = pending.content[chars_consumed_response:]
                    
                    if remaining_content:
                        pending_responses[0] = AgentResponse(content=remaining_content)
                    else:
                        pending_responses.pop(0)
                    
                    # Update remaining speech to only what wasn't spoken
                    remaining_speech = remaining_speech[chars_consumed_speech:]
                else:
                    # No match - skip this entire pending response
                    pending_responses.pop(0)
        
        return committed
